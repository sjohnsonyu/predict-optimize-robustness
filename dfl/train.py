import tensorflow as tf
import numpy as np
import argparse
import tqdm
import time
import sys
import pickle
import random
sys.path.insert(0, "../")

from dfl.model import ANN
from dfl.synthetic import generateDataset
from dfl.whittle import whittleIndex, newWhittleIndex
from dfl.utils import getSoftTopk, twoStageNLLLoss
from dfl.ope import opeIS, opeIS_parallel
from dfl.environments import POMDP2MDP

from armman.offline_trajectory import get_offline_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ARMMAN decision-focused learning')
    parser.add_argument('--method', default='TS', type=str, help='TS (two-stage learning) or DF (decision-focused learning).')
    parser.add_argument('--env', default='general', type=str, help='general (MDP) or POMDP.')
    parser.add_argument('--data', default='synthetic', type=str, help='synthetic or pilot')
    parser.add_argument('--sv', default='.', type=str, help='save string name')
    parser.add_argument('--epochs', default=50, type=int, help='num epochs')
    parser.add_argument('--instances', default=10, type=int, help='num instances')
    parser.add_argument('--ope', default='sim', type=str, help='importance sampling (IS) or simulation-based (sim).')
    parser.add_argument('--seed', default=0, type=int, help='random seed for synthetic data generation.')
    parser.add_argument('--noise_scale', default=0, type=float, help='sigma of normally random noise added to test set')
  

    args = parser.parse_args()
    print('argparser arguments', args)
    print ("OPE SETTING IS: ", args.ope)
    n_benefs = 100
    n_trials = 100
    L = 10
    K = 20
    n_states = 2
    gamma = 0.99
    target_policy_name = 'soft-whittle'
    beh_policy_name    = 'random'
    TS_WEIGHT=0.1
    noise_scale = args.noise_scale

    # Environment setup
    env = args.env
    H = 10
    seed = args.seed

    # Evaluation setup
    ope_mode = args.ope

    if args.data=='pilot':
        n_instances = 12
        all_n_benefs = 7668
        n_benefs = int(all_n_benefs/n_instances)
        n_benefs = 638
        n_trials = 1
        L = 7
        H = 7
        K = int(225/n_instances)
        n_states = 2
        gamma = 0.99
        full_dataset = get_offline_dataset(beh_policy_name, L, seed)
        single_trajectory = True
        # For offline data, seed must be set here
        np.random.seed(seed)
        random.seed(seed)
        tf.random.set_seed(seed)
    else:
        # dataset generation
        n_instances = args.instances
        # Seed are set inside generateDataset function
        train_val_dataset  = generateDataset(n_benefs, n_states, n_instances, n_trials, L, K, gamma, env=env, H=H, seed=seed)
        test_dataset  = generateDataset(n_benefs, n_states, n_instances, n_trials, L, K, gamma, env=env, H=H, seed=seed, dist_shift=True, noise_scale=noise_scale)
        single_trajectory = False


    train_dataset = train_val_dataset[:int(n_instances*0.8)]
    val_dataset   = train_val_dataset[int(n_instances*0.8):]
    # test_dataset  = test_dataset[int(n_instances*0.8):]

    dataset_list = [('train', train_dataset), ('val', val_dataset), ('test', test_dataset)]

    # model initialization
    model = ANN(n_states=n_states)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    loss_fn = tf.keras.losses.mean_squared_error

    # training
    training_mode = 'two-stage' if args.method == 'TS' else 'decision-focused'
    total_epoch = args.epochs
    overall_loss = {'train': [], 'test': [], 'val': []} # two-stage loss
    overall_ope = {'train': [], 'test': [], 'val': []} # OPE IS
    overall_ope_sim = {'train': [], 'test': [], 'val': []} # OPE simulation
    overall_ope_sim_regret = {'train': [], 'test': [], 'val': []}
    overall_ope_is_regret = {'train': [], 'test': [], 'val': []}
    for epoch in range(total_epoch+1):
        for mode, dataset in dataset_list:
            loss_list = []
            ope_list = [] # OPE IS
            ess_list = []
            ope_sim_list = [] # OPE simulation
            optimal_ope_is_regret_list = [] # OPE IS
            optimal_ope_sim_regret_list = [] # OPE simulation

            if mode == 'train':
                dataset = tqdm.tqdm(dataset)

            for (feature, ground_truth_T_data, raw_R_data, traj, ope_simulator, _, state_record, action_record, reward_record) in dataset:
                feature = tf.constant(feature, dtype=tf.float32)
                # label   = tf.constant(label, dtype=tf.float32)
                raw_R_data = tf.constant(raw_R_data, dtype=tf.float32)
                ground_truth_T_data = tf.constant(ground_truth_T_data, dtype=tf.float32)

                with tf.GradientTape() as tape:
                    prediction = model(feature) # Transition probabilities
                    # if epoch==total_epoch:
                    #     prediction=label
                    
                    # Setup MDP or POMDP environment
                    if env=='general':
                        T_data, R_data = prediction, raw_R_data
                        n_full_states = n_states
                    elif env=='POMDP':
                        T_data, R_data = POMDP2MDP(prediction, raw_R_data, H)
                        n_full_states = n_states * H
                    
                    # start_time = time.time()
                    loss = twoStageNLLLoss(traj, T_data, beh_policy_name) # - twoStageNLLLoss(traj, label, beh_policy_name) # Two-stage custom NLL loss
                    # print('two stage loss time:', time.time() - start_time)
 
                    # general idea: use ground truth test set, calculate own whittle
                    # calculate rewards and then subtract actual in order to obtain regret

                    # Batch Whittle index computation
                    # start_time = time.time()
                    # w = whittleIndex(prediction)
                    w = newWhittleIndex(T_data, R_data)
                    w = tf.reshape(w, (n_benefs, n_full_states))
                    if epoch == total_epoch:
                        w = tf.zeros((n_benefs, n_full_states))
                    # print('Whittle index time:', time.time() - start_time)
                    
                    # start_time = time.time()
                    ope_IS, ess = opeIS_parallel(state_record, action_record, reward_record, w, n_benefs, L, K, n_trials, gamma,
                            target_policy_name, beh_policy_name, single_trajectory=single_trajectory)
                    ope_sim = ope_simulator(w, K)
                    # ope_sim = ope_simulator(tf.reshape(w, (n_benefs, n_full_states)))
                    if ope_mode == 'IS': # importance-sampling based OPE
                        ess_weight = 0 # no ess weight. Purely CWPDIS
                        ope = ope_IS # - ess_weight * tf.reduce_sum(1.0 / tf.math.sqrt(ess))
                    elif ope_mode == 'sim': # simulation-based OPE
                        ope = ope_sim
                    else:
                        raise NotImplementedError
                    # print('Evaluation time:', time.time() - start_time)

                    ts_weight = TS_WEIGHT
                    performance = -ope * (1 - ts_weight) + loss * ts_weight

                    if mode == 'test':
                        print('entering')
                        optimal_w = newWhittleIndex(ground_truth_T_data, R_data)
                        optimal_ope_IS, optimal_ess = opeIS_parallel(state_record, action_record, reward_record, optimal_w, n_benefs, L, K, n_trials, gamma,
                            target_policy_name, beh_policy_name, single_trajectory=single_trajectory)
                        optimal_ope_sim = ope_simulator(optimal_w, K)
                        optimal_ope_sim_regret_list.append(optimal_ope_sim - ope_sim)
                        optimal_ope_is_regret_list.append(optimal_ope_IS - ope_IS)



                # backpropagation
                if mode == 'train' and epoch<total_epoch and epoch>0:
                    if training_mode == 'two-stage':
                        grad = tape.gradient(loss, model.trainable_variables)
                    elif training_mode == 'decision-focused':
                        grad = tape.gradient(performance, model.trainable_variables)
                    else:
                        raise NotImplementedError
                    optimizer.apply_gradients(zip(grad, model.trainable_variables))

                loss_list.append(loss)
                ope_list.append(ope_IS)
                ope_sim_list.append(ope_sim)
                ess_list.append(tf.reduce_mean(ess))

            print(f'Epoch {epoch}, {mode} mode, average loss {np.mean(loss_list)}, average ope (IS) {np.mean(ope_list)}, average ope (sim) {np.mean(ope_sim_list)}, average ess {np.mean(ess_list)}')
            if mode == 'test':
                print(f'average ope (IS) regret {np.mean(optimal_ope_is_regret_list)}, average ope (sim) regret {np.mean(optimal_ope_sim_regret_list)}')
            # import pdb; pdb.set_trace()
            overall_loss[mode].append(np.mean(loss_list))
            overall_ope[mode].append(np.mean(ope_list))
            overall_ope_sim[mode].append(np.mean(ope_sim_list))
            if mode == 'test':
                overall_ope_sim_regret[mode].append(np.mean(optimal_ope_sim_regret_list))
                overall_ope_is_regret[mode].append(np.mean(optimal_ope_is_regret_list))

    
    if not(args.sv == '.'):
        ### Output to be saved, else do nothing. 
        with open(args.sv, 'wb') as filename:
            pickle.dump([overall_loss, overall_ope, overall_ope_sim, overall_ope_is_regret, overall_ope_sim_regret], filename)


