import tensorflow as tf
import numpy as np
import tqdm

from dfl.policy import getActionProb, getActionProbNaive, getProbs
from dfl.config import dim_dict, policy_map
from dfl.trajectory import getSimulatedTrajectories
from dfl.trajectory import getEmpProbBenefLookup, getEmpProbClusterLookup, augmentTraj
from dfl.utils import aux_dict_to_transition_matrix

def opeIS(traj, w, mask, n_benefs, T, K, n_trials, gamma, target_policy_name, beh_policy_name):
    compare = {'target':policy_map[target_policy_name], 'beh':policy_map[beh_policy_name]}
    gamma_series = np.array([gamma**t for t in range(T-1)]) # Kai edited: it was **t-1** instead of **t**.

    beh_probs    = np.zeros((n_trials, T, n_benefs))
    target_probs = np.zeros((n_trials, T, n_benefs))

    v = []
    w_mask = tf.gather(w, mask) if tf.is_tensor(w) else w[mask] # Added the tensorflow version to support tensorflow indexing
    for benef in tqdm.tqdm(range(n_benefs), desc='OPE'):
        v_i = 0
        for trial in range(n_trials):
            imp_weight = 1
            v_i_tau = 0
            for ts in range(T-1):
                a_i_t = traj[trial, # trial index
                                compare['beh'], # policy index
                                ts, # time index
                                dim_dict['action'], # tuple dimension
                                benef # benef index
                                ].astype(int)

                s_t = traj[trial, # trial index
                                compare['beh'], # policy index
                                ts, # time index
                                dim_dict['state'], # tuple dimension
                                : # benef index
                                ].astype(int)
                pi_tar = getActionProb(s_t, a_i_t,
                                           policy=compare['target'],
                                           benef=benef, ts=ts,
                                           w=w_mask, k=K, N=n_benefs)
                pi_beh = getActionProb(s_t, a_i_t,
                                           policy=compare['beh'],
                                           benef=benef, ts=ts,
                                           w=w_mask, k=K, N=n_benefs)
                imp_weight*= pi_tar/pi_beh
                # if imp_weight>1:
                #     print('weight: ', imp_weight)
                v_i_t_tau = gamma_series[ts] * traj[trial, # trial index
                                                compare['beh'], # policy index
                                                ts, # time index
                                                dim_dict['reward'], # tuple dimension
                                                benef # benef index
                                                ] * imp_weight
                v_i_tau += v_i_t_tau

                beh_probs[trial, ts, benef]    = pi_beh
                target_probs[trial, ts, benef] = pi_tar

            v_i += v_i_tau
        v.append(v_i/n_trials)
    ope = np.sum(v)
    # print(f'OPE: {ope}')
    return ope

#This is the parallelized implementation of the same OPE. Ideally these two should match but the parallelized version is faster.
def opeIS_parallel(state_record, action_record, reward_record, w, mask, n_benefs, T, K, n_trials, gamma, target_policy_name, beh_policy_name):
    compare = {'target':policy_map[target_policy_name], 'beh':policy_map[beh_policy_name]}
    gamma_series = np.array([gamma**t for t in range(T-1)])

    ntr, _, L, N = state_record.shape

    v = []
    w_mask = tf.gather(w, mask) if tf.is_tensor(w) else w[mask] # Added the tensorflow version to support tensorflow indexing

    # state_record_beh = np.concatenate([np.tile(np.arange(N), (ntr, L, 1)).reshape(ntr, L, N, 1), state_record[:,compare['beh'],:,:].reshape(ntr, L, N, 1)], axis=-1).astype(int)
    action_record_beh = action_record[:,compare['beh'],:,:]

    # Get the corresponding Whittle indices
    # whittle_indices = tf.gather_nd(w, state_record_beh)

    # Batch topk to get probabilities
    beh_probs_raw    = tf.reshape(getProbs(state_record[:,compare['beh'],:,:].reshape(-1, N), policy=compare['beh'], ts=None, w=w_mask, k=K),    (ntr, L, N))
    target_probs_raw = tf.reshape(getProbs(state_record[:,compare['beh'],:,:].reshape(-1, N), policy=compare['target'], ts=None, w=w_mask, k=K), (ntr, L, N))

    # Use action to select the corresponding probabilities
    beh_probs    = beh_probs_raw * action_record_beh + (1 - beh_probs_raw) * (1 - action_record_beh)
    target_probs = target_probs_raw * action_record_beh + (1 - target_probs_raw) * (1 - action_record_beh)

    # Importance sampling weights
    IS_weights = target_probs / beh_probs # [ntr, L, N]

    # OPE
    total_probs = np.ones((ntr, N))
    ope = 0
    for t in range(T-1):
        rewards = reward_record[:, compare['beh'], t, :] # state_record[:,compare['beh'],t,:] # current state
        total_probs = total_probs * IS_weights[:,t,:]
        ope += rewards * total_probs * gamma_series[t]

    ope = tf.reduce_sum(ope) / ntr
    return ope

def opeISNaive(traj, w, mask, n_benefs, T, K, n_trials, gamma, target_policy_name, beh_policy_name):
    compare = {'target':policy_map[target_policy_name], 'beh':policy_map[beh_policy_name]}
    gamma_series = np.array([gamma**(t-1) for t in range(T-1)])

    v = []
    for trial in range(n_trials):
        imp_weight = 1
        v_tau = 0
        for ts in range(T-1):
            a_t = traj[trial, # trial index
                            compare['beh'], # policy index
                            ts, # time index
                            dim_dict['action'], # tuple dimension
                            : # benef index
                            ].astype(int)
            # a_t_encoded = encode_vector(a_t, N_ACTIONS)

            s_t = traj[trial, # trial index
                            compare['beh'], # policy index
                            ts, # time index
                            dim_dict['state'], # tuple dimension
                            : # benef index
                            ].astype(int)
            # s_t_encoded = encode_vector(s_t, N_STATES)

            pi_tar = getActionProbNaive(s_t, a_t, policy=compare['target'],
                                        w=w[mask], k=K, N=n_benefs)
            pi_beh = getActionProbNaive(s_t, a_t, policy=compare['beh'],
                                        w=w[mask], k=K, N=n_benefs)

            imp_weight*= pi_tar/pi_beh
            # if imp_weight>1:
            #     print('weight: ', imp_weight)
            v_t_tau = gamma_series[ts] * traj[trial, # trial index
                                            compare['beh'], # policy index
                                            ts, # time index
                                            dim_dict['reward'], # tuple dimension
                                            : # benef index
                                            ].sum() * imp_weight
            v_tau += v_t_tau
  
    v.append(v_tau)
    ope = np.mean(v)
    print(f'OPE Naive: {ope}')
    return ope

# Simulation-based OPE (differentiable and parallelizable)
class opeSimulation(object): # TODO
    def __init__(self, beh_traj, mask_seed, n_benefs, T, K, OPE_sim_n_trials, gamma, beh_policy_name):
        self.mask_seed = mask_seed
        self.n_benefs = n_benef
        self.T = T
        self.K = K
        self.OPE_sim_n_trials = OPE_sim_n_trials
        self.gamma = gamma

        policy_id = policy_map[beh_policy_name]
        emp_prob_by_benef_ssa, tr_df_benef_ssa, aux_dict_ssa = getEmpProbBenefLookup(beh_traj, policy_id, trial_id, n_benefs, True)
        self.est_T_data = aux_dict_to_transition_matrix(aux_dict_ssa, n_benefs)

    def __call__(self, w):
        compute = tf.custom_gradient(lambda x: self._compute(x))
        return compute(w)

    def _compute(self, w):
        traj, OPE_sim_whittle, simulated_rewards, mask, state_record, action_record, reward_record = getSimulatedTrajectories(
                                                    self.n_benefs, self.T, self.K, self.OPE_sim_n_trials, gamma,
                                                    sim_seed, mask_seed, self.est_T_data, w
                                                    )

        def gradient_function(dsoln):
            return grad_w

