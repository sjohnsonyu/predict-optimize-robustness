import subprocess
import sys
import os
import numpy as np
import itertools
import pickle
import matplotlib.pyplot as plt
import argparse


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


args=parser.parse_args()

args.save=bool(args.save)
args.plot=bool(args.plot)
args.compute=bool(args.compute)

if not args.name == '.':
  save_name = args.name
  print ("Using special save string: ", save_name)

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
    # print ('Starting DF Importance Sampling to be saved as: '+DF_IS_filename)
    # subprocess.run(f'python3 train.py --method DF --sv {DF_IS_filename} --epochs {args.epochs} --instances {args.instances} --seed {sd} --ope {"IS"} --noise_scale {args.noise_scale} {robust_clause}', shell=True)
    print ('Starting DF Simu based to be saved as:', df_sim_filename)
    subprocess.run(f'python3 {curr_dir}/train.py --method DF --sv {df_sim_filename} --epochs {args.epochs} --instances {args.instances} --seed {sd} --ope {"sim"} --noise_scale {args.noise_scale} {robust_clause} --adversarial {args.adversarial}', shell=True)
    print ('Starting TS to be saved as:', ts_filename)
    subprocess.run(f'python3 {curr_dir}/train.py --method TS --sv {ts_filename} --epochs {args.epochs} --instances {args.instances} --seed {sd} --noise_scale {args.noise_scale} {robust_clause} --adversarial {args.adversarial}', shell=True)
    print ('BOTH DONE')


if args.plot:
  ### Plot figures for the specified settings if True 
  modes = ['train', 'val', 'test']
  for mode in modes:
    # df_is_outputs = []
    df_sim_outputs = []
    ts_outputs = []
    for sd in range(args.seed, args.seed + args.tr):

      # DF_IS_filename='./results/DF_IS_'+special+'_sd_'+str(sd)+'.pickle'
      df_sim_filename = f'./results/DF_SIM_{save_name}_sd_{sd}.pickle'
      ts_filename = f'./results/TS_{save_name}_sd_{sd}.pickle'

      # with open (DF_IS_filename, 'rb') as df_is_file:
      #     df_is_outputs.append(pickle.load(df_is_file))

      with open (df_sim_filename, 'rb') as df_sim_file:
          df_sim_outputs.append(pickle.load(df_sim_file))

      with open (ts_filename, 'rb') as ts_file:
          ts_outputs.append(pickle.load(ts_file))


    num_epochs = len(df_sim_outputs[0][0][mode])# - 1 ## Last entry is the OPE if GT is perfectly known

    random_metrics = [[ts_outputs[sd-args.seed][i][mode][0] for i in range(3)] for sd in range(args.seed, args.seed+args.tr)]
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
        df_sim_outputs_for_this_epoch = np.array([item[0][mode][epoch] for item in df_sim_outputs])  # loss
        ts_outputs_for_this_epoch = np.array([item[0][mode][epoch] for item in ts_outputs])

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
    
    for epoch in range(num_epochs):
        df_sim_outputs_for_this_epoch = np.array([item[1][mode][epoch] for item in df_sim_outputs])
        ts_outputs_for_this_epoch = np.array([item[1][mode][epoch] for item in ts_outputs])
        optimal_outputs_for_this_epoch = np.array([item[2][mode][epoch] for item in df_sim_outputs])
        df_sim_means.append(np.mean(df_sim_outputs_for_this_epoch))
        ts_means.append(np.mean(ts_outputs_for_this_epoch))
        optimal_means.append(np.mean(optimal_outputs_for_this_epoch))
        df_sim_errors.append(np.std(df_sim_outputs_for_this_epoch)/np.sqrt(len(df_sim_outputs_for_this_epoch)))
        ts_errors.append(np.std(ts_outputs_for_this_epoch)/np.sqrt(len(ts_outputs_for_this_epoch)))
        optimal_errors.append(np.std(optimal_outputs_for_this_epoch)/np.sqrt(len(optimal_outputs_for_this_epoch)))
    
    df_sim_means = np.array(df_sim_means)
    df_sim_errors = np.array(df_sim_errors)
    ts_means = np.array(ts_means)
    ts_errors = np.array(ts_errors)
    optimal_means = np.array(optimal_means)
    optimal_errors = np.array(optimal_errors)
    
    plt.plot(range(num_epochs), df_sim_means, label='DF-SIM')
    plt.fill_between(range(num_epochs), df_sim_means-df_sim_errors, df_sim_means+df_sim_errors, alpha=0.2)
    
    plt.plot(range(num_epochs), ts_means, label='TS', color='#F4B400', lw=lw)
    plt.fill_between(range(num_epochs), ts_means-ts_errors, ts_means+ts_errors, alpha=0.2, color='#F4B400')
    
    plt.plot(range(num_epochs), optimal_means, label='optimal', color='#53AD58', lw=lw)
    plt.fill_between(range(num_epochs), optimal_means-optimal_errors, optimal_means+optimal_errors, alpha=0.2, color='#53AD58')

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
  random_regrets = []
  ts_regrets = []
  df_regrets = []
  optimal_opes = []
  for sd in range(args.seed, args.seed+args.tr):
    # Optimal selected epoch
    optimal_selected_epoch = np.argmax(df_sim_outputs[sd-args.seed][2]['val'][:-1]) # Maximize SIM OPE
    optimal_selected_metrics.append([0, df_sim_outputs[sd-args.seed][2]['test'][optimal_selected_epoch]])
    optimal_performance = df_sim_outputs[sd-args.seed][2]['test'][optimal_selected_epoch]
    optimal_opes.append(optimal_performance)

    # Two-stage selected epoch
    print ('seed:', sd, 'out of', len(ts_outputs))
    # ts_selected_epoch = np.argmin(ts_outputs[sd-args.seed][0]['val'][:-1]) # loss metric
    ts_selected_epoch = np.argmax(ts_outputs[sd-args.seed][1]['val'][:-1]) # Maximize SIM OPE
    ts_selected_metrics.append([ts_outputs[sd-args.seed][i]['test'][ts_selected_epoch] for i in range(2)])
    ts_regrets.append(optimal_performance - ts_outputs[sd-args.seed][1]['test'][ts_selected_epoch])

    # DF-sim selected epoch
    df_sim_selected_epoch = np.argmax(df_sim_outputs[sd-args.seed][1]['val'][:-1]) # Maximize SIM OPE
    df_sim_selected_metrics.append([df_sim_outputs[sd-args.seed][i]['test'][df_sim_selected_epoch] for i in range(2)])
    df_regrets.append(optimal_performance - df_sim_outputs[sd-args.seed][1]['test'][df_sim_selected_epoch])


  ts_selected_metrics = np.array(ts_selected_metrics)
  optimal_selected_metrics = np.array(optimal_selected_metrics)
  df_sim_selected_metrics = np.array(df_sim_selected_metrics)

  ts_test_mean, ts_test_ste = np.mean(ts_selected_metrics, axis=0), np.std(ts_selected_metrics, axis=0) / np.sqrt(len(ts_outputs))
  optimal_test_mean, optimal_test_ste = np.mean(optimal_selected_metrics, axis=0), np.std(optimal_selected_metrics, axis=0) / np.sqrt(len(ts_outputs))
  df_sim_test_mean, df_sim_test_ste = np.mean(df_sim_selected_metrics, axis=0), np.std(df_sim_selected_metrics, axis=0) / np.sqrt(len(df_sim_outputs))

  print(f'Random test metrics mean (Loss | Sim OPE):       {random_mean[0]:.3f}\t{random_mean[1]:.1f}')
  print(f'Two-stage test metrics mean (Loss | Sim OPE):    {ts_test_mean[0]:.3f}\t{ts_test_mean[1]:.1f}')
  print(f'DF-sim test metrics mean (Loss | Sim OPE):       {df_sim_test_mean[0]:.3f}\t{df_sim_test_mean[1]:.1f}')
  print(f'Optimal DF test metrics mean (Loss | Sim OPE):   N/A\t{optimal_test_mean[1]:.1f}')



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
  rewards = [random_mean[1], ts_test_mean[1], df_sim_test_mean[1], optimal_test_mean[1]]
  reward_errors = [random_ste[1], ts_test_ste[1], df_sim_test_ste[1], optimal_test_ste[1]]

  plt.bar(labels, rewards, yerr=reward_errors, capsize=5, color=colors)
  plt.title("Reward (Sim-OPE)")
  plt.ylabel("Reward")

  if args.save:
    plt.savefig(f'./figs/{save_name}_rewards_bar.png')
  # plt.show()

  
