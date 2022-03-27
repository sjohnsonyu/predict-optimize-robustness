import tensorflow as tf
import numpy as np
import argparse
import tqdm
import time
import sys
import os
import pickle
import random
sys.path.insert(0, "../")

from dfl.model import ANN
from dfl.trajectory import getSimulatedTrajectories
from dfl.synthetic import generateDataset
from dfl.whittle import whittleIndex, newWhittleIndex
from dfl.utils import getSoftTopk, twoStageNLLLoss, euclideanLoss
from dfl.ope import opeIS_parallel, opeSimulator
from dfl.utils import addRandomNoise, flipClusterProbabilities
from dfl.environments import POMDP2MDP

from armman.offline_trajectory import get_offline_dataset

# TODO put into a constants file
OPE_SIM_N_TRIALS = 100
# OPTIMAL_EPSILON = 0.01
OPTIMAL_EPSILON = 0.05
print('using epsilon =', OPTIMAL_EPSILON)

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
    parser.add_argument('--robust', default=None, type=str, help='method of robust training')

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
    elif args.data == 'synthetic':
        # dataset generation
        n_instances = args.instances
        # Seed is set inside generateDataset function
        full_dataset  = generateDataset(n_benefs, n_states, n_instances, n_trials, L, K, gamma, env=env, H=H, seed=seed)
        single_trajectory = False
    else:
        raise NotImplementedError

    train_dataset = full_dataset[:int(n_instances*0.7)]
    val_dataset   = full_dataset[int(n_instances*0.7):int(n_instances*0.8)]
    test_dataset  = full_dataset[int(n_instances*0.8):]
    dataset_list = [('train', train_dataset), ('val', val_dataset), ('test', test_dataset)]

    # model initialization
    model = ANN(n_states=n_states)
    model.build((None, train_dataset[0][0].shape[1]))
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    loss_fn = tf.keras.losses.mean_squared_error

    # Model list (visualization only)
    model_list = []

    # training
    training_mode = 'two-stage' if args.method == 'TS' else 'decision-focused'
    total_epoch = args.epochs
    overall_loss = {'train': [], 'test': [], 'val': []} # two-stage loss
    overall_ope_sim = {'train': [], 'test': [], 'val': []} # OPE simulation
    overall_ope_sim_optimal = {'train': [], 'test': [], 'val': []}
    for epoch in range(total_epoch+1):
        model_list.append(model.get_weights())
        for mode, dataset in dataset_list:
            loss_list = []
            ess_list = []
            ope_sim_list = [] # OPE simulation
            ope_sim_optimal_list = [] # OPE simulation
            if mode == 'train':
                dataset = tqdm.tqdm(dataset)

            for (feature, label, raw_R_data, traj, ope_simulator, _, state_record, action_record, reward_record) in dataset:
                feature = tf.constant(feature, dtype=tf.float32)
                raw_R_data = tf.constant(raw_R_data, dtype=tf.float32)

                # ================== computing optimal solution ===================
                if args.data == 'synthetic':
                    label = tf.constant(label, dtype=tf.float32)
                    T_data, R_data = label, raw_R_data
                    n_full_states = n_states

                    if mode == 'test' or args.robust == 'add_noise':
                        label = addRandomNoise(label, noise_scale)
                        
                        ope_simulator = opeSimulator(None, n_benefs, L, n_states, OPE_SIM_N_TRIALS, gamma, beh_policy_name='random', T_data=label.numpy(), R_data=R_data.numpy(), env=env, H=H)

                    w_optimal = newWhittleIndex(label, R_data)
                    w_optimal = tf.reshape(w_optimal, (n_benefs, n_full_states))
                    optimal_loss = euclideanLoss(T_data, label)
                    # optimal_loss = twoStageNLLLoss(traj, label, beh_policy_name)
                    ope_sim_optimal = ope_simulator(w_optimal, K, epsilon=OPTIMAL_EPSILON)

                else: # no label available in the pilot dataset
                    w_optimal = tf.zeros((n_benefs, n_full_states)) # random
                    optimal_loss = 0
                    ope_sim_optimal = 0

                # ======================= Tracking gradient ========================
                with tf.GradientTape() as tape:
                    prediction = model(feature) # Transition probabilities

                    # Setup MDP environment
                    T_data, R_data = prediction, raw_R_data
                    n_full_states = n_states
                    
                    # loss = twoStageNLLLoss(traj, T_data, beh_policy_name) - optimal_loss
                    loss = euclideanLoss(T_data, label) - optimal_loss

                    # Batch Whittle index computation
                    w = newWhittleIndex(T_data, R_data)
                    w = tf.reshape(w, (n_benefs, n_full_states))

                    if epoch == total_epoch:
                        w = tf.zeros((n_benefs, n_full_states))

                    # evaluation_epsilon = 1.0 / 10 if mode == 'train' else 0.01  # TODO: do we want to change between train and test?
                    evaluation_epsilon = OPTIMAL_EPSILON
                    ope_sim = ope_simulator(w, K, epsilon=evaluation_epsilon)

                    performance = -ope_sim * (1 - TS_WEIGHT) + loss * TS_WEIGHT

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
                ope_sim_list.append(ope_sim)
                ope_sim_optimal_list.append(ope_sim_optimal)

            print(f'Epoch {epoch}, {mode} mode, average loss {np.mean(loss_list):.4f}, ' +
                    f'average ope (sim) {np.mean(ope_sim_list):.2f}, ' +
                    f'optimal ope (sim) {np.mean(ope_sim_optimal_list):.2f}')

            overall_loss[mode].append(np.mean(loss_list))
            overall_ope_sim[mode].append(np.mean(ope_sim_list))
            overall_ope_sim_optimal[mode].append(np.mean(ope_sim_optimal_list))

    folder_path = 'pretrained/{}'.format(args.data)

    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    model_path = '{}/{}.pickle'.format(folder_path, args.method)
    with open(model_path, 'wb') as f:
        pickle.dump((train_dataset, val_dataset, test_dataset, model_list), f)

    if not(args.sv == '.'):
        ### Output to be saved, else do nothing. 
        with open(args.sv, 'wb') as filename:
            pickle.dump([overall_loss, overall_ope_sim, overall_ope_sim_optimal], filename)
            # pickle.dump([overall_loss, overall_ope, overall_ope_sim, overall_ope_IS_optimal, overall_ope_sim_optimal], filename)


