import subprocess
import sys
import os
import numpy as np
import itertools
import pickle
import matplotlib.pyplot as plt
import argparse
from collections import defaultdict


parser=argparse.ArgumentParser(description = 'Visulation and tuning DF vs TS comparison')
# parser.add_argument('--epochs', default=10, type=int, help='Number of epochs')
parser.add_argument('--seed', default=0, type=int, help='Random seed')
parser.add_argument('--tr', default=40, type=int, help='Number of trials to be run starting with seed value entered for seed.')
# parser.add_argument('--name', default='.', type=str, help='Special string name.')


args=parser.parse_args()


results_dir = '/Users/sonja/Downloads/dfl/results_created/'
base_filename = '-3_5_eps_0.03_gamma_0.9'

NOISE_AMOUNTS = {'adversarial': ['baseline_rewards', 'window_0.1', 'window_0.2', 'window_0.3'],
'adversarial_training': ['baseline_rewards', 'window_0.1', 'window_0.2', 'window_0.3'],
'random': ['baseline_rewards', 'window_0.1_rewards', 'window_0.2_rewards', 'window_0.3_rewards']
}
EXP_TYPES = ['adversarial', 'adversarial_training', 'random']


diff_summaries = {'adversarial': [], 'adversarial_training': [], 'random': []}

df_ope_means = {'adversarial': defaultdict(dict), 'adversarial_training': defaultdict(dict), 'random': defaultdict(dict)}
ts_ope_means = {'adversarial': defaultdict(dict), 'adversarial_training': defaultdict(dict), 'random': defaultdict(dict)}

# experiment type --> noise amount --> mode --> seed
df_sim_outputs = {'adversarial': defaultdict(list), 'adversarial_training': defaultdict(list), 'random': defaultdict(list)}
ts_outputs = {'adversarial': defaultdict(list), 'adversarial_training': defaultdict(list), 'random': defaultdict(list)}

allowed_seeds_template = [defaultdict(list), defaultdict(list), defaultdict(list)]

ts_allowed_seeds = {'adversarial': dict(zip(NOISE_AMOUNTS['adversarial'], [defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)])),
                   'adversarial_training': dict(zip(NOISE_AMOUNTS['adversarial_training'], [defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)])),
                   'random': dict(zip(NOISE_AMOUNTS['random'], [defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)]))
                   }
df_allowed_seeds = {'adversarial': dict(zip(NOISE_AMOUNTS['adversarial'], [defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)])),
                   'adversarial_training': dict(zip(NOISE_AMOUNTS['adversarial_training'], [defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)])),
                   'random': dict(zip(NOISE_AMOUNTS['random'], [defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)]))
                   }
# ts_allowed_seeds = {'adversarial': defaultdict(list), 'adversarial_training': defaultdict(list), 'random': defaultdict(list)}
# df_allowed_seeds = {'adversarial': defaultdict(list), 'adversarial_training': defaultdict(list), 'random': defaultdict(list)}
both_allowed_seeds = {'adversarial': defaultdict(list), 'adversarial_training': defaultdict(list), 'random': defaultdict(list)}

io_output = []

for exp_type in EXP_TYPES:
  for noise_amount in NOISE_AMOUNTS[exp_type]:
    for mode in ['train', 'val', 'test']:
      for sd in range(args.seed, args.seed + args.tr):
        # Load files and keep track of valid seeds
        save_name = f'{noise_amount}_{base_filename}_{exp_type}'
        df_sim_filename = f'{results_dir}DF_SIM_{save_name}_sd_{sd}.pickle'
        ts_filename = f'{results_dir}TS_{save_name}_sd_{sd}.pickle'
        io_output.append(df_sim_filename)

        # if sd != 0:
        #   breakpoint()

        both_exist = True
        try:
          with open (df_sim_filename, 'rb') as df_sim_file:
              loaded_df_sim_file = pickle.load(df_sim_file)
              df_allowed_seeds[exp_type][noise_amount][mode].append(sd)
        except:
          io_output.append(f'{df_sim_filename} does not exist')
          # print(f'{df_sim_filename} does not exist')
          both_exist = False
        try:
          with open (ts_filename, 'rb') as ts_file:
              loaded_ts_file = pickle.load(ts_file)
              ts_allowed_seeds[exp_type][noise_amount][mode].append(sd)
        except:
          io_output.append(f'{ts_filename} does not exist')
          # print(f'{ts_filename} does not exist')
          both_exist = False

        # FIXME: what about MODE? should I only accept seeds that are valid for ALL modes?
        if both_exist:
          df_sim_outputs[exp_type][noise_amount].append(loaded_df_sim_file)
          ts_outputs[exp_type][noise_amount].append(loaded_ts_file)
          both_allowed_seeds[exp_type][noise_amount].append(sd)
      
      # TODO: consider disaggregating allowed seeds for TS and DF
      if len(df_sim_outputs[exp_type][noise_amount]) == 0:
        print('skipping', df_sim_filename)
        breakpoint()
        continue

      # if len(df_sim_outputs[exp_type][noise_amount]) > 0:


      num_epochs = len(df_sim_outputs[exp_type][noise_amount][0][0][mode])# - 1 ## Last entry is the OPE if GT is perfectly known
      ts_num_epochs = len(ts_outputs[exp_type][noise_amount][0][0][mode])# - 1 ## Last entry is the OPE if GT is perfectly known
      df_num_epochs = num_epochs
      io_output.append(f'{exp_type} {noise_amount} {mode} successful ts trials: {len(ts_outputs[exp_type][noise_amount][0][0][mode])}')
      io_output.append(f'{exp_type} {noise_amount} {mode} successful df trials: {len(df_sim_outputs[exp_type][noise_amount][0][0][mode])}')
      # print(f'{exp_type} {noise_amount} {mode} successful ts trials:', len(ts_outputs[exp_type][noise_amount][0][0][mode]))
      # print(f'{exp_type} {noise_amount} {mode} successful df trials:', len(df_sim_outputs[exp_type][noise_amount][0][0][mode]))
      # if ts_num_epochs < 2 or df_num_epochs < 2:
        # breakpoint()
        # continue
        # TODO check this — might need to roll back if test doesn't have them but val does

      df_sim_means = []
      ts_means = []
      
      for epoch in range(num_epochs):
        df_sim_outputs_for_this_epoch = np.array([item[1][mode][epoch] for item in df_sim_outputs[exp_type][noise_amount] if len(item[1][mode]) > epoch])
        ts_outputs_for_this_epoch = np.array([item[1][mode][epoch] for item in ts_outputs[exp_type][noise_amount] if len(item[1][mode]) > epoch])
        df_sim_means.append(np.mean(df_sim_outputs_for_this_epoch))
        ts_means.append(np.mean(ts_outputs_for_this_epoch))
      
      df_sim_means = np.array(df_sim_means)
      ts_means = np.array(ts_means)
      df_ope_means[exp_type][noise_amount][mode] = df_sim_means
      ts_ope_means[exp_type][noise_amount][mode] = ts_means


    ts_selected_metrics = []
    df_sim_selected_metrics = []
    for sd_idx, sd in enumerate(both_allowed_seeds[exp_type][noise_amount]):
      # Two-stage selected epoch
      if len(ts_outputs[exp_type][noise_amount][sd_idx][1]['val'][:-1]) > 0:
        ts_selected_epoch = np.argmax(ts_outputs[exp_type][noise_amount][sd_idx][1]['val'][:-1]) # Maximize SIM OPE
        ts_selected_metrics.append(ts_outputs[exp_type][noise_amount][sd_idx][1]['test'][ts_selected_epoch])

      # # DF-sim selected epoch
      if len(df_sim_outputs[exp_type][noise_amount][sd_idx][1]['val'][:-1]) > 0:
        df_sim_selected_epoch = np.argmax(df_sim_outputs[exp_type][noise_amount][sd_idx][1]['val'][:-1]) # Maximize SIM OPE
        df_sim_selected_metrics.append(df_sim_outputs[exp_type][noise_amount][sd_idx][1]['test'][df_sim_selected_epoch])
  
    diff_summaries[exp_type].append(np.mean(ts_selected_metrics) - np.mean(df_sim_selected_metrics))  # starts with baseline


with open('io_output.txt', 'w') as f:
  newlined_output = [line + '\n' for line in io_output]
  f.writelines(newlined_output)

plt.figure()
for exp_type in EXP_TYPES:
  plt.plot(np.linspace(0, 0.3, 4), diff_summaries[exp_type], label=exp_type)
plt.title("TS OPE - DF OPE")
plt.xlabel("Noise window")
plt.ylabel("TS OPE Reward - DF OPE Reward")
plt.legend()
plt.savefig("noise_vs_difference.png")
plt.show()