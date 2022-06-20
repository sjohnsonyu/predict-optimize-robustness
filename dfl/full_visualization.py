import subprocess
import sys
import os
import numpy as np
import itertools
import pickle
import matplotlib.pyplot as plt
import argparse

OPTIMAL_EPSILON = 0.1

parser=argparse.ArgumentParser(description = 'Visulation and tuning DF vs TS comparison')
parser.add_argument('--epochs', default=10, type=int, help='Number of epochs')
parser.add_argument('--instances', default=10, type=int, help='Number of instances')
parser.add_argument('--seed', default=0, type=int, help='Random seed')
parser.add_argument('--save', default=0, type=int, help='Whether or not to save all generated figs. Put 0 for False, 1 for True')
parser.add_argument('--plot', default=0, type=int, help='Whether or not to create plots. Put 0 for False, 1 for True')
parser.add_argument('--compute', default=0, type=int, help='Whether or not to run new experiments. Put 0 for False, 1 for True')
parser.add_argument('--tr', default=1, type=int, help='Number of trials to be run starting with seed value entered for seed.')
parser.add_argument('--name', default='.', type=str, help='Special string name.')
parser.add_argument('--noise_scale', default=0, type=float, help='sigma of normally random noise added to test set')
parser.add_argument('--robust', default=None, type=str, help='method of robust training')
parser.add_argument('--adversarial', default=0, type=int, help='0 if using random perturb, 1 if adversarial')
parser.add_argument('--eps', default=OPTIMAL_EPSILON, type=float, help='epsilon used for calculating soft top k')


args=parser.parse_args()

args.save=bool(args.save)
args.plot=bool(args.plot)
args.compute=bool(args.compute)

if not args.name == '.':
  save_name = args.name
  # print ("Using special save string: ", save_name)

if args.compute:
  ### Launch new computational experiments for the specified settings if True 
  for sd in range(args.seed, args.seed+args.tr):
 
    # DF_IS_filename='./results/DF_IS_'+special+'_sd_'+str(sd)+'.pickle'
    curr_dir = os.path.abspath(os.getcwd())
    df_sim_filename = f'{curr_dir}/results/DF_SIM_{save_name}_sd_{sd}.pickle'
    ts_filename = f'{curr_dir}/results/TS_{save_name}_sd_{sd}.pickle'
    # df_sim_filename = f'./results/DF_SIM_{save_name}_sd_{sd}.pickle'
    # ts_filename = f'./results/TS_{save_name}_sd_{sd}.pickle'

    robust_clause = '' if not args.robust == 'add_noise' else '--robust add_noise'
    print ('Starting seed: ', sd)
    print ('Starting DF Simu based to be saved as:', df_sim_filename)
    subprocess.run(f'python3 {curr_dir}/train.py --method DF --sv {df_sim_filename} --epochs {args.epochs} --instances {args.instances} --seed {sd} --ope {"sim"} --noise_scale {args.noise_scale} {robust_clause} --adversarial {args.adversarial} --eps {args.eps}', shell=True)
    print ('Starting TS to be saved as:', ts_filename)
    subprocess.run(f'python3 {curr_dir}/train.py --method TS --sv {ts_filename} --epochs {args.epochs} --instances {args.instances} --seed {sd} --noise_scale {args.noise_scale} {robust_clause} --adversarial {args.adversarial} --eps {args.eps}', shell=True)
    print ('BOTH DONE')


if args.plot:
  ### Plot figures for the specified settings if True 
  modes = ['train', 'val', 'test']
  for mode in modes:
    # df_is_outputs = []
    df_sim_outputs = []
    ts_outputs = []
    ts_allowed_seeds = []
    df_allowed_seeds = []
    both_allowed_seeds = []
    do_nothing_allowed_seeds = []
    do_nothing_outputs = []
    random_actions_allowed_seeds = []
    random_actions_outputs = []
    random_on_disengaged_allowed_seeds = []
    random_on_disengaged_outputs = []
    lowest_rewards_allowed_seeds = []
    lowest_rewards_outputs = []
    random_on_low_rewards_allowed_seeds = []
    random_on_low_rewards_outputs = []
    for sd in range(args.seed, args.seed + args.tr):
      results_dir = '/Users/sonja/Downloads/dfl/random_training/'
      # results_dir = '/Users/sonja/Downloads/dfl/results_20220531/results/'
      # DF_IS_filename='./results/DF_IS_'+special+'_sd_'+str(sd)+'.pickle'
      df_sim_filename = f'{results_dir}DF_SIM_{save_name}_sd_{sd}.pickle'
      ts_filename = f'{results_dir}TS_{save_name}_sd_{sd}.pickle'
      # do_nothing_filename = f'{results_dir}DO_NOTHING_{save_name}_sd_{sd}.pkl'
      # do_nothing_filename = f'{results_dir}DO_NOTHING_{save_name}_sd_{sd}.pkl'
      results_dir = '/Users/sonja/Downloads/dfl/results_20220531/results/'
      do_nothing_filename = f'{results_dir}DO_NOTHING_{save_name}_sd_{sd}.pkl'
      random_actions_filename = f'{results_dir}RANDOM_ACTIONS_{save_name}_sd_{sd}.pkl'
      random_on_disengaged_filename = f'{results_dir}random_on_disengaged_{save_name}_sd_{sd}.pkl'
      lowest_rewards_filename = f'{results_dir}lowest_rewards_{save_name}_sd_{sd}.pkl'
      random_on_low_rewards_filename = f'{results_dir}random_on_low_rewards_{save_name}_sd_{sd}.pkl'
      if 'training' in do_nothing_filename:
        training_start_idx = do_nothing_filename.find('training_')
        training_end_idx = training_start_idx + len('training_')
        do_nothing_filename = do_nothing_filename[:training_start_idx] + do_nothing_filename[training_end_idx:]
      if 'training' in random_actions_filename:
        training_start_idx = random_actions_filename.find('training_')
        training_end_idx = training_start_idx + len('training_')
        random_actions_filename = random_actions_filename[:training_start_idx] + random_actions_filename[training_end_idx:]
      if 'training' in random_on_disengaged_filename:
        training_start_idx = random_on_disengaged_filename.find('training_')
        training_end_idx = training_start_idx + len('training_')
        random_on_disengaged_filename = random_on_disengaged_filename[:training_start_idx] + random_on_disengaged_filename[training_end_idx:]
      if 'training' in lowest_rewards_filename:
        training_start_idx = lowest_rewards_filename.find('training_')
        training_end_idx = training_start_idx + len('training_')
        lowest_rewards_filename = lowest_rewards_filename[:training_start_idx] + lowest_rewards_filename[training_end_idx:]
      if 'training' in random_on_low_rewards_filename:
        training_start_idx = random_on_low_rewards_filename.find('training_')
        training_end_idx = training_start_idx + len('training_')
        random_on_low_rewards_filename = random_on_low_rewards_filename[:training_start_idx] + random_on_low_rewards_filename[training_end_idx:]
      
      if 'ts_weight_0.1' in do_nothing_filename:
        ts_weight_start_idx = do_nothing_filename.find('ts_weight_0.1_')
        ts_weight_end_idx = ts_weight_start_idx + len('ts_weight_0.1_')
        do_nothing_filename = do_nothing_filename[:ts_weight_start_idx] + do_nothing_filename[ts_weight_end_idx:]

        ts_weight_start_idx = random_actions_filename.find('ts_weight_0.1_')
        ts_weight_end_idx = ts_weight_start_idx + len('ts_weight_0.1_')
        random_actions_filename = random_actions_filename[:ts_weight_start_idx] + random_actions_filename[ts_weight_end_idx:]

        ts_weight_start_idx = random_on_disengaged_filename.find('ts_weight_0.1_')
        ts_weight_end_idx = ts_weight_start_idx + len('ts_weight_0.1_')
        random_on_disengaged_filename = random_on_disengaged_filename[:ts_weight_start_idx] + random_on_disengaged_filename[ts_weight_end_idx:]

        ts_weight_start_idx = lowest_rewards_filename.find('ts_weight_0.1_')
        ts_weight_end_idx = ts_weight_start_idx + len('ts_weight_0.1_')
        lowest_rewards_filename = lowest_rewards_filename[:ts_weight_start_idx] + lowest_rewards_filename[ts_weight_end_idx:]

        ts_weight_start_idx = random_on_low_rewards_filename.find('ts_weight_0.1_')
        ts_weight_end_idx = ts_weight_start_idx + len('ts_weight_0.1_')
        random_on_low_rewards_filename = random_on_low_rewards_filename[:ts_weight_start_idx] + random_on_low_rewards_filename[ts_weight_end_idx:]
      # df_sim_filename = f'./results_scp/DF_SIM_{save_name}_sd_{sd}.pickle'
      # ts_filename = f'./results_scp/TS_{save_name}_sd_{sd}.pickle'
      
      # with open (DF_IS_filename, 'rb') as df_is_file:
      #     df_is_outputs.append(pickle.load(df_is_file))
      both_exist = True
      try:
        with open (df_sim_filename, 'rb') as df_sim_file:
            loaded_df_sim_file = pickle.load(df_sim_file)
            if len(loaded_df_sim_file[0]['test']) < 2:
              both_exist = False
            # df_sim_outputs.append(pickle.load(df_sim_file))
        # df_allowed_seeds.append(sd)
      except:
        print(f'{df_sim_filename} does not exist')
        both_exist = False
      try:
        with open (ts_filename, 'rb') as ts_file:
            loaded_ts_file = pickle.load(ts_file)
            if len(loaded_ts_file[0]['test']) < 2:
              both_exist = False
            # ts_outputs.append(pickle.load(ts_file))
        # ts_allowed_seeds.append(sd)
      except:
        print(f'{ts_filename} does not exist')
        both_exist = False

      try:
        with open (do_nothing_filename, 'rb') as do_nothing_file:
            loaded_do_nothing_file = pickle.load(do_nothing_file)
            do_nothing_outputs.append(loaded_do_nothing_file)
        do_nothing_allowed_seeds.append(sd)
      except:
        print(f'{do_nothing_filename} does not exist')
        both_exist = False

      try:
        with open (random_actions_filename, 'rb') as random_actions_file:
            loaded_random_actions_file = pickle.load(random_actions_file)
            random_actions_outputs.append(loaded_random_actions_file)
        random_actions_allowed_seeds.append(sd)
      except:
        print(f'{random_actions_filename} does not exist')
        both_exist = False
      try:
        with open (random_on_disengaged_filename, 'rb') as random_on_disengaged_file:
            loaded_random_on_disengaged_file = pickle.load(random_on_disengaged_file)
            random_on_disengaged_outputs.append(loaded_random_on_disengaged_file)
        random_on_disengaged_allowed_seeds.append(sd)
      except:
        print(f'{random_on_disengaged_filename} does not exist')
        both_exist = False
      try:
        with open (lowest_rewards_filename, 'rb') as lowest_rewards_file:
            loaded_lowest_rewards_file = pickle.load(lowest_rewards_file)
            lowest_rewards_outputs.append(loaded_lowest_rewards_file)
        lowest_rewards_allowed_seeds.append(sd)
      except:
        print(f'{lowest_rewards_filename} does not exist')
        both_exist = False
      try:
        with open (random_on_low_rewards_filename, 'rb') as random_on_low_rewards_file:
            loaded_random_on_low_rewards_file = pickle.load(random_on_low_rewards_file)
            random_on_low_rewards_outputs.append(loaded_random_on_low_rewards_file)
        random_on_low_rewards_allowed_seeds.append(sd)
      except:
        print(f'{random_on_low_rewards_filename} does not exist')
        both_exist = False

      if both_exist:
        df_sim_outputs.append(loaded_df_sim_file)
        ts_outputs.append(loaded_ts_file)
        both_allowed_seeds.append(sd)

    num_epochs = len(df_sim_outputs[0][0][mode])# - 1 ## Last entry is the OPE if GT is perfectly known

    random_metrics = [[ts_outputs[j][i][mode][0] for i in range(3)] for j in range(len(both_allowed_seeds))]
    # random_metrics = [[ts_outputs[j][i][mode][0] for i in range(3)] for j in range(len(ts_allowed_seeds))]
    # random_metrics = [[ts_outputs[sd-args.seed][i][mode][0] for i in range(3)] for sd in range(args.seed, args.seed+args.tr)]
    random_mean, random_ste = np.mean(random_metrics, axis=0), np.std(random_metrics, axis=0) / np.sqrt(len(ts_outputs))
    ### Loss figure
    plt.figure()
    
    lw = 3

    df_sim_means = []
    df_sim_errors = []
    df_sim_regrets = []
    ts_means = []
    ts_errors = []
    
    
    for epoch in range(num_epochs):
      df_sim_outputs_for_this_epoch = np.array([item[0][mode][epoch] for item in df_sim_outputs if len(item[0][mode]) > epoch])  # loss
      ts_outputs_for_this_epoch = np.array([item[0][mode][epoch] for item in ts_outputs if len(item[0][mode]) > epoch])

      df_sim_means.append(np.mean(df_sim_outputs_for_this_epoch))  # loss at given epoch, across diff seeds
      ts_means.append(np.mean(ts_outputs_for_this_epoch))
      

      df_sim_errors.append(np.std(df_sim_outputs_for_this_epoch)/np.sqrt(len(df_sim_outputs_for_this_epoch)))
      ts_errors.append(np.std(ts_outputs_for_this_epoch)/np.sqrt(len(ts_outputs_for_this_epoch)))
    
    
    df_sim_means = np.array(df_sim_means)
    df_sim_errors = np.array(df_sim_errors)
    
    ts_means = np.array(ts_means)
    ts_errors = np.array(ts_errors)

    

    plt.plot(range(num_epochs), df_sim_means, label='DF-SIM')
    plt.fill_between(range(num_epochs), df_sim_means-df_sim_errors, df_sim_means+df_sim_errors, alpha=0.2)  # df loss per epoch
    
    plt.plot(range(num_epochs), ts_means, label='TS', color='#F4B400', lw=lw)
    plt.fill_between(range(num_epochs), ts_means-ts_errors, ts_means+ts_errors, alpha=0.2, color='#F4B400')  # ts loss per epoch

    plt.hlines(random_mean[0], xmin=0, xmax=num_epochs, colors='#DB4437', lw=lw, linestyle='dashed', label='random')

    plt.legend(fontsize=18)
    plt.xlabel('Epochs', fontsize=18)
    plt.ylabel('Intermediate Loss', fontsize=18)
    plt.title(mode+' Loss comparison', fontsize=18)
    if args.save:
        plt.savefig('./figs/'+save_name+'_'+mode+'_loss.png')
    # plt.show()

    # print('successful ts trials:', len(ts_outputs))
    # print('successful df trials:', len(df_sim_outputs))
    ### SIM-OPE figure
    plt.figure()
    
    df_is_means = []
    df_is_errors = []
    df_sim_means = []
    df_sim_errors = []
    ts_means = []
    ts_errors = []
    optimal_means = []
    optimal_errors = []

    ts_reward_data = []
    df_reward_data = []
    
    do_nothing_means = []
    do_nothing_errors = []
    random_actions_means = []
    random_actions_errors = []

    # breakpoint()
    for epoch in range(num_epochs):
        df_sim_outputs_for_this_epoch = np.array([item[1][mode][epoch] for item in df_sim_outputs if len(item[1][mode]) > epoch])
        ts_outputs_for_this_epoch = np.array([item[1][mode][epoch] for item in ts_outputs if len(item[1][mode]) > epoch])
        optimal_outputs_for_this_epoch = np.array([item[2][mode][epoch] for item in df_sim_outputs if len(item[1][mode]) > epoch])
        df_sim_means.append(np.mean(df_sim_outputs_for_this_epoch))
        ts_means.append(np.mean(ts_outputs_for_this_epoch))
        optimal_means.append(np.mean(optimal_outputs_for_this_epoch))
        df_sim_errors.append(np.std(df_sim_outputs_for_this_epoch)/np.sqrt(len(df_sim_outputs_for_this_epoch)))
        ts_errors.append(np.std(ts_outputs_for_this_epoch)/np.sqrt(len(ts_outputs_for_this_epoch)))
        optimal_errors.append(np.std(optimal_outputs_for_this_epoch)/np.sqrt(len(optimal_outputs_for_this_epoch)))
        df_reward_data.append(df_sim_outputs_for_this_epoch)
        ts_reward_data.append(ts_outputs_for_this_epoch)

        do_nothing_outputs_for_this_epoch = np.array([item[1][mode][epoch] for item in do_nothing_outputs if len(item[1][mode]) > epoch])
        do_nothing_means.append(np.mean(do_nothing_outputs_for_this_epoch))
        do_nothing_errors.append(np.std(do_nothing_outputs_for_this_epoch)/np.sqrt(len(do_nothing_outputs_for_this_epoch)))

        random_actions_outputs_for_this_epoch = np.array([item[1][mode][epoch] for item in random_actions_outputs if len(item[1][mode]) > epoch])
        random_actions_means.append(np.mean(random_actions_outputs_for_this_epoch))
        random_actions_errors.append(np.std(random_actions_outputs_for_this_epoch)/np.sqrt(len(random_actions_outputs_for_this_epoch)))

    # import pandas as pd
    # plt.figure(figsize=(10, 6))
    # df_reward_data_df = pd.DataFrame(df_reward_data)
    # c = 'blue'
    # plt.boxplot(df_reward_data_df)
    # breakpoint()
    # # plt.boxplot(df_reward_data_df, positions=np.arange(len(df_reward_data_df)), widths=0.25, patch_artist=True,
    # #         boxprops=dict(facecolor=c, color=c),
    # #         capprops=dict(color=c),
    # #         whiskerprops=dict(color=c),
    # #         flierprops=dict(color=c, markeredgecolor=c),
    # #         medianprops=dict(linewidth=2))
    # plt.title(f'DF Rewards by Epoch - {mode}')
    # plt.xlabel('Epoch')
    # plt.ylabel('OPE SIM Rewards')
    # plt.savefig('./figs/'+save_name+'_'+mode+'_df_reward_by_epoch.png')
    # plt.figure()
    # # plt.show()
    # plt.title(f'TS Rewards by Epoch - {mode}')
    # plt.xlabel('Epoch')
    # plt.ylabel('OPE SIM Rewards')
    # ts_reward_data_df = pd.DataFrame(ts_reward_data)
    # c = 'green'
    # plt.boxplot(ts_reward_data_df)
    # # plt.boxplot(ts_reward_data_df, positions=np.arange(len(df_reward_data_df)) + 0.5, widths=0.25, patch_artist=True,
    # #         boxprops=dict(facecolor=c, color=c),
    # #         capprops=dict(color=c),
    # #         whiskerprops=dict(color=c),
    # #         flierprops=dict(color=c, markeredgecolor=c),
    # #         medianprops=dict(linewidth=2))
    # plt.xticks(np.arange(0, 21))
    # plt.savefig('./figs/'+save_name+'_'+mode+'_ts_reward_by_epoch.png')

    df_sim_means = np.array(df_sim_means)
    df_sim_errors = np.array(df_sim_errors)
    ts_means = np.array(ts_means)
    ts_errors = np.array(ts_errors)
    optimal_means = np.array(optimal_means)
    optimal_errors = np.array(optimal_errors)
    do_nothing_means = np.array(do_nothing_means)
    do_nothing_errors = np.array(do_nothing_errors)
    random_actions_means = np.array(random_actions_means)
    random_actions_errors = np.array(random_actions_errors)
    
    plt.figure()
    plt.plot(range(num_epochs), df_sim_means - do_nothing_means, label='DF-SIM')
    plt.fill_between(range(num_epochs), df_sim_means-do_nothing_means-df_sim_errors, df_sim_means-do_nothing_means+df_sim_errors, alpha=0.2)
    
    plt.plot(range(num_epochs), ts_means-do_nothing_means, label='TS', color='#F4B400', lw=lw)
    plt.fill_between(range(num_epochs), ts_means-do_nothing_means-ts_errors, ts_means-do_nothing_means+ts_errors, alpha=0.2, color='#F4B400')
    
    plt.plot(range(num_epochs), optimal_means, label='optimal', color='#53AD58', lw=lw)
    plt.fill_between(range(num_epochs), optimal_means-optimal_errors, optimal_means+optimal_errors, alpha=0.2, color='#53AD58')

    # plt.plot(range(num_epochs), df_sim_means, label='DF-SIM')
    # plt.fill_between(range(num_epochs), df_sim_means-df_sim_errors, df_sim_means+df_sim_errors, alpha=0.2)
    
    # plt.plot(range(num_epochs), ts_means, label='TS', color='#F4B400', lw=lw)
    # plt.fill_between(range(num_epochs), ts_means-ts_errors, ts_means+ts_errors, alpha=0.2, color='#F4B400')
    
    # plt.plot(range(num_epochs), optimal_means, label='optimal', color='#53AD58', lw=lw)
    # plt.fill_between(range(num_epochs), optimal_means-optimal_errors, optimal_means+optimal_errors, alpha=0.2, color='#53AD58')

    plt.hlines(random_mean[1], xmin=0, xmax=num_epochs, colors='#DB4437', lw=lw, linestyle='dashed', label='random')
    
    plt.legend(fontsize=18)
    plt.xlabel('Epochs', fontsize=18)
    plt.ylabel('OPE-Sim', fontsize=18)
    plt.title(mode+' Sim-OPE comparison', fontsize=18)
    if args.save:
        plt.savefig('./figs/'+save_name+'_'+mode+'_OPE_SIM.png')
    # plt.show()
  

  ### Table information
  ts_selected_metrics = []
  df_sim_selected_metrics = []
  optimal_selected_metrics = []
  do_nothing_selected_metrics = []
  random_actions_selected_metrics = []
  random_on_disengaged_selected_metrics = []
  lowest_rewards_selected_metrics = []
  random_on_low_rewards_selected_metrics = []
  random_regrets = []
  ts_regrets = []
  df_regrets = []
  optimal_opes = []
  do_nothing_opes = []
  random_actions_opes = []
  for sd_idx, sd in enumerate(both_allowed_seeds):
    # breakpoint()
    # Optimal selected epoch
    # TODO try minimizing regret, if that's possible for selecting metrics?
    # breakpoint()
    optimal_selected_epoch = np.argmax(df_sim_outputs[sd_idx][2]['val'][:-1]) # Maximize SIM OPE
    optimal_selected_metrics.append([0, df_sim_outputs[sd_idx][2]['test'][optimal_selected_epoch]])
    optimal_performance = df_sim_outputs[sd_idx][2]['test'][optimal_selected_epoch]
    optimal_opes.append(optimal_performance)

    # Do nothing selected epoch
    do_nothing_selected_epoch = np.argmax(do_nothing_outputs[sd_idx][1]['val'][:-1]) # Maximize SIM OPE
    do_nothing_selected_metrics.append([0, do_nothing_outputs[sd_idx][1]['test'][do_nothing_selected_epoch]])
    do_nothing_performance = do_nothing_outputs[sd_idx][1]['test'][do_nothing_selected_epoch]
    do_nothing_opes.append(do_nothing_performance)

    # Random actions selected epoch
    random_actions_selected_epoch = np.argmax(random_actions_outputs[sd_idx][1]['val'][:-1]) # Maximize SIM OPE
    random_actions_selected_metrics.append([0, random_actions_outputs[sd_idx][1]['test'][random_actions_selected_epoch]])
    random_actions_performance = random_actions_outputs[sd_idx][1]['test'][random_actions_selected_epoch]
    random_actions_opes.append(random_actions_performance)
    # Random actions selected epoch
    random_on_disengaged_selected_epoch = np.argmax(random_on_disengaged_outputs[sd_idx][1]['val'][:-1]) # Maximize SIM OPE
    random_on_disengaged_selected_metrics.append([0, random_on_disengaged_outputs[sd_idx][1]['test'][random_on_disengaged_selected_epoch]])
    # Random actions selected epoch
    lowest_rewards_selected_epoch = np.argmax(lowest_rewards_outputs[sd_idx][1]['val'][:-1]) # Maximize SIM OPE
    lowest_rewards_selected_metrics.append([0, lowest_rewards_outputs[sd_idx][1]['test'][lowest_rewards_selected_epoch]])
    # Random actions selected epoch
    random_on_low_rewards_selected_epoch = np.argmax(random_on_low_rewards_outputs[sd_idx][1]['val'][:-1]) # Maximize SIM OPE
    random_on_low_rewards_selected_metrics.append([0, random_on_low_rewards_outputs[sd_idx][1]['test'][random_on_low_rewards_selected_epoch]])

    # Two-stage selected epoch
    # print ('seed:', sd, 'out of', len(ts_outputs))
    val_ts_reward = np.array(ts_outputs[sd_idx][1]['val'][1:-1])
    val_ts_opt_reward = np.array(ts_outputs[sd_idx][2]['val'][1:-1])
    ts_selected_epoch = np.argmin(val_ts_opt_reward - val_ts_reward) + 1
    # print('ts val regrets', val_ts_opt_reward - val_ts_reward)
    # print('ts val rewards', val_ts_reward)
    # print('ts val opt rewards', val_ts_opt_reward)
    # ts_selected_epoch = np.argmax(ts_outputs[sd_idx][1]['val'][:-1]) # Maximize SIM OPE
    ts_selected_metrics.append([ts_outputs[sd_idx][i]['test'][ts_selected_epoch] for i in range(2)])
    ts_regrets.append(ts_outputs[sd_idx][2]['test'][ts_selected_epoch] - ts_outputs[sd_idx][1]['test'][ts_selected_epoch])
    # print('picked ts', ts_selected_epoch)

    # # DF-sim selected epoch
    val_df_sim_reward = np.array(df_sim_outputs[sd_idx][1]['val'][1:-1])
    val_df_sim_opt_reward = np.array(df_sim_outputs[sd_idx][2]['val'][1:-1])
    if len(val_df_sim_reward) == 0: continue
    df_sim_selected_epoch = np.argmin(val_df_sim_opt_reward - val_df_sim_reward) + 1
    # print('df val regrets', val_df_sim_opt_reward - val_df_sim_reward)
    # print('df val rewards', val_df_sim_reward)
    # print('df val opt rewards', val_df_sim_opt_reward)
    # print('picked df', df_sim_selected_epoch)
    # df_sim_selected_epoch = np.argmax(df_sim_outputs[sd_idx][1]['val'][:-1]) # Maximize SIM OPE
    df_sim_selected_metrics.append([df_sim_outputs[sd_idx][i]['test'][df_sim_selected_epoch] for i in range(2)])
    df_regrets.append(df_sim_outputs[sd_idx][2]['test'][df_sim_selected_epoch] - df_sim_outputs[sd_idx][1]['test'][df_sim_selected_epoch])
  
  plt.figure()
  data = [ts_regrets, df_regrets]
  plt.title("Regret (Sim-OPE)")
  plt.ylabel("Regret")
  plt.boxplot(data)
  plt.xticks([1, 2], ['TS', 'DF'])
  plt.axhline(y=0, color='gray', linestyle='--')
  plt.savefig(f'./figs/{save_name}_regrets_boxplot.png')

  ts_selected_metrics = np.array(ts_selected_metrics)
  optimal_selected_metrics = np.array(optimal_selected_metrics)
  df_sim_selected_metrics = np.array(df_sim_selected_metrics)
  do_nothing_selected_metrics = np.array(do_nothing_selected_metrics)
  random_actions_selected_metrics = np.array(random_actions_selected_metrics)
  random_on_disengaged_selected_metrics = np.array(random_on_disengaged_selected_metrics)
  lowest_rewards_selected_metrics = np.array(lowest_rewards_selected_metrics)
  random_on_low_rewards_selected_metrics = np.array(random_on_low_rewards_selected_metrics)
  temp = args.name
  first_idx = temp.find('_')
  second_idx = temp.find('_', first_idx+1)
  name = temp[:second_idx]
  import pandas as pd
  df = pd.read_csv('stats_train_noise.csv', index_col=0)
  df[f'{name}_ts'] = ts_selected_metrics[:,1]
  df[f'{name}_df'] = df_sim_selected_metrics[:,1]
  df[f'{name}_baseline'] = do_nothing_selected_metrics[:,1]
  df.to_csv('stats_train_noise.csv')


  ts_test_mean, ts_test_ste = np.mean(ts_selected_metrics, axis=0), np.std(ts_selected_metrics, axis=0) / np.sqrt(len(ts_outputs))
  optimal_test_mean, optimal_test_ste = np.mean(optimal_selected_metrics, axis=0), np.std(optimal_selected_metrics, axis=0) / np.sqrt(len(ts_outputs))
  df_sim_test_mean, df_sim_test_ste = np.mean(df_sim_selected_metrics, axis=0), np.std(df_sim_selected_metrics, axis=0) / np.sqrt(len(df_sim_outputs))
  do_nothing_test_mean, do_nothing_test_ste = np.mean(do_nothing_selected_metrics, axis=0), np.std(do_nothing_selected_metrics, axis=0) / np.sqrt(len(do_nothing_outputs))
  random_actions_test_mean, random_actions_test_ste = np.mean(random_actions_selected_metrics, axis=0), np.std(random_actions_selected_metrics, axis=0) / np.sqrt(len(random_actions_outputs))
  random_on_disengaged_test_mean, random_on_disengaged_test_ste = np.mean(random_on_disengaged_selected_metrics, axis=0), np.std(random_on_disengaged_selected_metrics, axis=0) / np.sqrt(len(random_on_disengaged_outputs))
  lowest_rewards_test_mean, lowest_rewards_test_ste = np.mean(lowest_rewards_selected_metrics, axis=0), np.std(lowest_rewards_selected_metrics, axis=0) / np.sqrt(len(lowest_rewards_outputs))
  random_on_low_rewards_test_mean, random_on_low_rewards_test_ste = np.mean(random_on_low_rewards_selected_metrics, axis=0), np.std(random_on_low_rewards_selected_metrics, axis=0) / np.sqrt(len(random_on_low_rewards_outputs))

  # print(f'Random test metrics mean (Loss | Sim OPE):       {random_mean[0]:.3f}\t{random_mean[1]:.1f}')
  # print(f'Two-stage test metrics mean (Loss | Sim OPE):    {ts_test_mean[0]:.3f}\t{ts_test_mean[1]:.1f}')
  # print(f'DF-sim test metrics mean (Loss | Sim OPE):       {df_sim_test_mean[0]:.3f}\t{df_sim_test_mean[1]:.1f}')
  # print(f'Do nothing test metrics mean (Loss | Sim OPE):       {do_nothing_test_mean[0]:.3f}\t{do_nothing_test_mean[1]:.1f}')
  # print(f'Optimal DF test metrics mean (Loss | Sim OPE):   N/A\t{optimal_test_mean[1]:.1f}')

  # print(f'{random_mean[1]} {ts_test_mean[1]} {df_sim_test_mean[1]} {do_nothing_test_mean[1]}')
  print(f'{ts_test_mean[1]} {df_sim_test_mean[1]} {do_nothing_test_mean[1]} {do_nothing_test_mean[1]} {random_actions_test_mean[1]} {random_actions_test_mean[1]} {random_on_disengaged_test_mean[1]} {random_on_disengaged_test_mean[1]} {lowest_rewards_test_mean[1]} {lowest_rewards_test_mean[1]} {random_on_low_rewards_test_mean[1]} {random_on_low_rewards_test_mean[1]}')

  labels = ['random', 'ts', 'df-sim', 'optimal']
  colors = ['#DB4437', '#F4B400', '#1f77b4', '#53AD58']
  plt.figure()
  losses = [random_mean[0], ts_test_mean[0], df_sim_test_mean[0]]
  loss_errors = [random_ste[0], ts_test_ste[0], df_sim_test_ste[0]]
  plt.bar(labels[:-1], losses, yerr=loss_errors, capsize=5, color=colors[:-1])
  plt.title("Loss")
  plt.ylabel("Loss")
  # plt.show()
  if args.save:
    plt.savefig(f'./figs/{save_name}_losses_bar.png')

  ts_regret_mean, ts_regret_ste = np.mean(ts_regrets, axis=0), np.std(ts_regrets, axis=0) / np.sqrt(len(ts_outputs))
  # optimal_regret_mean, optimal_regret_ste = np.mean(optimal_regret, axis=0), np.std(optimal_regret, axis=0) / np.sqrt(len(ts_outputs))
  df_regret_mean, df_regret_ste = np.mean(df_regrets, axis=0), np.std(df_regrets, axis=0) / np.sqrt(len(df_sim_outputs))
  random_regret = np.array(optimal_opes) - random_mean[1]
  random_regret_mean, random_regret_ste = np.mean(random_regret, axis=0), np.std(random_regret, axis=0) / np.sqrt(len(df_sim_outputs))
  plt.figure()

  regrets = [random_regret_mean, ts_regret_mean, df_regret_mean]
  regret_errors = [random_regret_ste, ts_regret_ste, df_regret_ste]
  plt.bar(labels[:-1], regrets, yerr=regret_errors, capsize=5, color=colors[:-1])
  plt.title("Regret (Sim-OPE)")
  plt.ylabel("Regret")
  if args.save:
    plt.savefig(f'./figs/{save_name}_regrets_bar.png')

  plt.figure()
  rewards = [random_mean[1]-do_nothing_test_mean[1], ts_test_mean[1]-do_nothing_test_mean[1], df_sim_test_mean[1]-do_nothing_test_mean[1]]
  reward_errors = [random_ste[1], ts_test_ste[1], df_sim_test_ste[1]]
  # rewards = [random_mean[1]-do_nothing_test_mean[1], ts_test_mean[1]-do_nothing_test_mean[1], df_sim_test_mean[1]-do_nothing_test_mean[1], optimal_test_mean[1]-do_nothing_test_mean[1]]
  # reward_errors = [random_ste[1], ts_test_ste[1], df_sim_test_ste[1], optimal_test_ste[1]]

  plt.bar(labels[:-1], rewards, yerr=reward_errors, capsize=5, color=colors[:-1])
  plt.title("Reward (Sim-OPE)")
  plt.ylabel("Reward")
  plt.ylim(0, 500)
  if args.save:
    plt.savefig(f'./figs/{save_name}_rewards_bar.png')
  # plt.show()

  
