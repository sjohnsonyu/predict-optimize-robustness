import sys
import os
import numpy as np
import itertools
import pickle
import matplotlib.pyplot as plt
import argparse
from copy import deepcopy

SAVE_DIR = '/Users/sonja/Downloads/dfl/results_created/'


def parse_for_df_or_ts(f, line, losses, rewards, opt_rewards):
    while not "sv='" in line and line != '':
        line = f.readline()
    if line == '': return -1, -1
    df_sv_start_idx = line.find('/n/home05/sjohnsonyu/predict-optimize-robustness/dfl/results/') + len('/n/home05/sjohnsonyu/predict-optimize-robustness/dfl/results/')
    df_sv_path = SAVE_DIR + line[df_sv_start_idx:-3]
    if '.pickle' not in df_sv_path: return -1, 1

    while not "Epoch" in line and line != '':
        line = f.readline()
    if line == '': return -1, -1

    while "Epoch" in line and line != '':
        epoch_start_idx = line.find("Epoch ") + len("Epoch ")
        epoch_end_idx = line.find(',', epoch_start_idx)
        epoch_num = int(line[epoch_start_idx:epoch_end_idx])
        
        mode_start_idx = line.find(' ', epoch_end_idx) + 1
        mode_end_idx = line.find(' ', mode_start_idx)
        mode = line[mode_start_idx:mode_end_idx]

        loss_start_idx = line.find('loss ') + len('loss ')
        loss_end_idx = line.find(',', loss_start_idx)
        loss = float(line[loss_start_idx:loss_end_idx])

        reward_start_idx = line.find('average ope (sim) ') + len('average ope (sim) ')
        reward_end_idx = line.find(',', reward_start_idx)
        reward = float(line[reward_start_idx:reward_end_idx])

        opt_reward_start_idx = line.find('optimal ope (sim) ') + len('optimal ope (sim) ')
        opt_reward_end_idx = line.find('\n', opt_reward_start_idx)
        opt_reward = float(line[opt_reward_start_idx:opt_reward_end_idx])

        losses[mode].append(loss)
        rewards[mode].append(reward)
        opt_rewards[mode].append(opt_reward)

        line = f.readline()
    if line == '': return -1, -1

    with open(df_sv_path, 'wb') as f:
        pickle.dump([losses, rewards, opt_rewards], f)

    return f, line


def parse_file(filename):
    out_template = {'train': [], 'val': [], 'test': []}

    with open(filename) as f:
        line = f.readline()
        _, line = parse_for_df_or_ts(f, line, deepcopy(out_template), deepcopy(out_template), deepcopy(out_template))
        if line != -1:
            parse_for_df_or_ts(f, line, deepcopy(out_template), deepcopy(out_template), deepcopy(out_template))


def parse_all_files_in_dir(dirname):
    successes = 0
    attempts = 0
    for file in os.listdir(dirname):
        if file.endswith(".out"):
            try:
                filename = os.path.join(dirname, file)
                parse_file(filename)
                # print(file, 'was good to go!')
                successes += 1
            except Exception as e:
                print(file, 'had a problem')
                print(e)
            attempts += 1
    print(successes/attempts)
# parse_file('/Users/sonja/Downloads/dfl/10931304_output.out')

parse_all_files_in_dir('/Users/sonja/Downloads/dfl/')