import tensorflow as tf
import numpy as np
from armman.simulator import getTopk
from dfl.config import dim_dict, policy_map
from dfl.utils import getSoftTopk, DiffTopK

WHITTLE_EPS = 1e-2

def getActions(states, policy, ts, w, k):
    N = len(states)
    actions=np.zeros(N)
    if policy == 0:
        # Select k arms at random
        actions[np.random.choice(np.arange(len(states)), k)] = 1
    elif policy == 1:
        # Select k arms in round robin
        actions[[(ts*k+i)%N for i in range(k)]] = 1
    elif policy == 2:
        # select k arms by Whittle
        N = len(states)
        actions=np.zeros(N)
        whittle_indices=w[np.arange(N), states]

        top_k_whittle=getTopk(whittle_indices, k)
        actions[top_k_whittle]=1
    elif policy == 3:
        # select k arms by Whittle using soft top k
        N = len(states)
        actions=np.zeros(N)
        whittle_indices=w[np.arange(N), states]

        soft_top_k_whittle=getSoftTopk([whittle_indices], k)
        actions[soft_top_k_whittle]=1

    return actions


def getActionProbDefault(states, action, policy):
    return 0.5

def getActionProbFrequentist(traj, state, action, benef, ts, policy, min_support = 1):
    s_t = traj[:, # trial index
        policy, # policy index
        ts, # time index
        dim_dict['state'], # tuple dimension
        : # benef index
        ]
    state_match_bool = (np.expand_dims(state, axis=0) == s_t).sum(axis=1)==\
                        traj.shape[4] 
    action_match_bool = traj[:, # trial index
        policy, # policy index
        ts, # time index
        dim_dict['action'], # tuple dimension
        benef # benef index
        ] == action
    if state_match_bool.sum()<min_support:
        return 0
        return getActionProbDefault(state, action, policy)
    else:
        prob = (state_match_bool & action_match_bool).sum()/state_match_bool.sum()
        return prob
    
def getActionProb(states, action, policy, benef, ts=None, w=None, k=None, N=None):
    if policy==policy_map['random']:
        return getActionProbRandom(states, action, k, N)
    elif policy==policy_map['rr']:
        return getActionProbRandom(states, action, k, N)
    elif policy==policy_map['whittle']:
        return getActionProbWhittle(states, action, benef, w, k, N)
    elif policy==policy_map['soft-whittle']:
        return getActionProbSoftWhittle(states, action, benef, w, k, N)
    else:
        raise f'Policy {policy} not supported'
    
def getActionProbWhittle(states, action, benef, w, k, N):
    ts=None
#     N = len(states)
    whittle_actions = getActions(states, policy_map['whittle'], ts, w, k)

    if whittle_actions[benef] == action: # Match
        if action == 1: # True action 1, observed action 1
            return 1 - WHITTLE_EPS
        else:           # True action 0, observed action 0
            return 1 - WHITTLE_EPS * k / (N-k)
    else: # Not match
        if action == 1: # True action 0, observed action 1
            return WHITTLE_EPS * k / (N-k)
        else:           # True action 1, observed action 0
            return WHITTLE_EPS

    # Kai: this part seems buggy!! # TODO
    # if whittle_actions[benef]==action:
    #     return 1-WHITTLE_EPS
    # else:
    #     return WHITTLE_EPS*k/(N-k)

def getActionProbSoftWhittle(states, action, benef, w, k, N):
    n = len(w)
    diffTopK = DiffTopK(k=k)
    w_selected = tf.reshape(tf.gather_nd(w, list(zip(range(n), states))), (1,n))
    gamma = diffTopK(-w_selected)
    probs = gamma[0,:,0].numpy() * n
    return probs[benef] * action + (1 - probs[benef]) * (1 - action)

def getProbSoftWhittle(states, benef, w, k, N):
    n = len(w)
    diffTopK = DiffTopK(k=k)
    w_selected = tf.reshape(tf.gather_nd(w, list(zip(range(n), states))), (1,n))
    gamma = diffTopK(-w_selected)
    probs = gamma[0,:,0].numpy() * n
    return probs
    
def getActionProbWhittleNaive(states, action, w, k):
    WHITTLE_EPS = 1e-2
    ts=None
    whittle_actions = getActions(states, policy_map['whittle'], ts, w, k)
    if np.array_equal(whittle_actions, action):
        return 1-WHITTLE_EPS
    else:
        return WHITTLE_EPS
    
def getActionProbRandom(states, action, k, N):
    ## select k arms according to whittle indices
    N = len(states)
#     return nck(N-1, k-1)/nck(N, k)
    if action:
        return k/N
    else:
        return 1-k/N

# Batch version
def getProbs(states, policy, ts, w, k):
    bs, N = states.shape
    if policy == 0:
        probs = tf.ones((bs, N)) * k / N

    elif policy == 1:
        # raise NotImplementedError('Round robin is not Markovian')
        # Pretend round robin to be random.
        probs = tf.ones((bs, N)) * k / N

    elif policy == 2: # Strict top k
        default_indices = tf.tile(tf.reshape(tf.range(N), (1,N)), (bs,1))
        states = tf.cast(states, dtype=tf.int32)
        states_with_indices = tf.concat([tf.expand_dims(default_indices, axis=-1), tf.expand_dims(states, axis=-1)], axis=-1)
        # print('strict top k')
        # print('state shape', states_with_indices.shape) # [bs, N, 2]

        w_selected = tf.gather_nd(w, states_with_indices)
        # print('whittle shape', w_selected.shape) # [bs, N]

        values, indices = tf.math.top_k(w_selected, k=k)
        # print('indices shape', indices.shape)

        probs = np.ones((bs, N)) * WHITTLE_EPS * k / (N-k)
        for i in range(bs):
            probs[i,indices[i]] = 1 - WHITTLE_EPS
        
        probs = tf.convert_to_tensor(probs, dtype=tf.float32)

        # print('probs shape', probs.shape)

    elif policy == 3: # Soft top k
        default_indices = tf.tile(tf.reshape(tf.range(N), (1,N)), (bs,1))
        states = tf.cast(states, dtype=tf.int32)
        states_with_indices = tf.concat([tf.expand_dims(default_indices, axis=-1), tf.expand_dims(states, axis=-1)], axis=-1)
        # print('soft top k')
        # print('state shape', states_with_indices.shape) # [bs, N, 2]

        w_selected = tf.gather_nd(w, states_with_indices)
        # print('whittle shape', w_selected.shape) # [bs, N]

        diffTopK = DiffTopK(k=k)
        gamma = diffTopK(-w_selected)
        probs = gamma[:,:,0] * N

        # print('probs shape', probs.shape)

    return probs
