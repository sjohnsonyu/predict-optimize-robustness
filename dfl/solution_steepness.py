import subprocess
import sys
import os
import numpy as np
import itertools
import pickle
import matplotlib.pyplot as plt
import argparse
from collections import defaultdict
import json
from eps_success_regret import get_avg_df_regret

WINDOW_SIZES = np.linspace(0, 0.5, 11)
SEEDS = np.arange(0, 40)


def get_avg_df_regret(window):  # TODO: combine this into one function with the one from eps_success_regret (and maybe DF too)
    df_sim_outputs = []
    save_name = f'window_{window}_-3_5_eps_0.04_gamma_0.9_adversarial_fast_window'
    results_dir = '~/Downloads/dfl/results/'
    for seed in SEEDS:
        df_sim_filename = f'{results_dir}DF_SIM_{save_name}_sd_{seed}.pickle'  # TODO update save_name and dir
        try:
            with open (df_sim_filename, 'rb') as df_sim_file:
                loaded_df_sim_file = pickle.load(df_sim_file)
                df_sim_outputs.append(loaded_df_sim_file)
        except:
            pass
    df_sim_selected_metrics = []
    df_regrets = []
    for i in range(len(df_sim_outputs)):
        optimal_selected_epoch = np.argmax(df_sim_outputs[i][2]['val'][:-1]) # Maximize SIM OPE
        optimal_performance = df_sim_outputs[i][2]['test'][optimal_selected_epoch]
        df_sim_selected_epoch = np.argmax(df_sim_outputs[i][1]['val'][:-1]) # Maximize SIM OPE
        df_regrets.append(optimal_performance - df_sim_outputs[i][1]['test'][df_sim_selected_epoch])

    if not df_regrets:
        breakpoint()
    df_regret_mean, df_regret_ste = np.mean(df_regrets, axis=0), np.std(df_regrets, axis=0) / np.sqrt(len(df_sim_outputs))
    return df_regret_mean, df_regret_ste

    
def get_avg_ts_regret(window):  # TODO: combine this into one function with the one from eps_success_regret (and maybe DF too)
    ts_outputs = []
    save_name = f'SOMETHING_{window}'
    results_dir = '~/dfl/results/'
    for seed in SEEDS:
        ts_filename = f'{results_dir}TS_{save_name}_sd_{seed}.pickle'  # TODO update save_name and dir
        try:
            with open (ts_filename, 'rb') as ts_file:
                loaded_ts_file = pickle.load(ts_file)
                ts_outputs.append(loaded_ts_file)
        except:
            pass
    ts_selected_metrics = []
    ts_regrets = []
    for i in range(len(ts_outputs)):
        optimal_selected_epoch = np.argmax(ts_outputs[i][2]['val'][:-1]) # Maximize SIM OPE
        optimal_performance = ts_outputs[i][2]['test'][optimal_selected_epoch]
        ts_selected_epoch = np.argmax(ts_outputs[i][1]['val'][:-1]) # Maximize SIM OPE
        ts_regrets.append(optimal_performance - ts_outputs[i][1]['test'][ts_selected_epoch])

    if not ts_regrets:
        breakpoint()
    ts_regret_mean, ts_regret_ste = np.mean(ts_regrets, axis=0), np.std(ts_regrets, axis=0) / np.sqrt(len(ts_outputs))
    return ts_regret_mean, ts_regret_ste

# NOTE: design idea -- function that gets the correct name! that should fix the diff function problem.


def plot(ts_regrets, ts_stes, df_regrets, df_stes):
    plt.figure()
    color1 = 'tab:blue'
    color2 = 'tab:red'
    plt.ylabel('Avg Regret (of Best Epoch)')
    plt.xlabel('Window Size')
    plt.title('Window Size vs. Regret')
    plt.plot(WINDOW_SIZES, ts_regrets, color=color1, label='TS')
    plt.fill_between(WINDOW_SIZES, ts_regrets-ts_stes, ts_regrets+ts_stes, color=color1, alpha=0.2)
    plt.plot(WINDOW_SIZES, df_regrets, color=color2, label='DF')
    plt.fill_between(WINDOW_SIZES, df_regrets-df_stes, df_regrets+df_stes, color=color2, alpha=0.2)
    plt.legend()
    plt.savefig('solution_steepness.png')
    plt.show()


def main():
    ts_regrets = []
    ts_stes = []
    df_regrets = []
    df_stes = []

    for window in WINDOW_SIZES:
        df_regret, df_ste = get_avg_df_regret(window)
        ts_regret, ts_ste = get_avg_ts_regret(window)
        ts_regrets.append(ts_regret)
        ts_stes.append(ts_ste)
        df_regrets.append(df_regret)
        df_stes.append(df_ste)

    ts_regrets = np.array(ts_regrets)
    ts_stes = np.array(ts_stes)
    df_regrets = np.array(df_regrets)
    df_stes = np.array(df_stes)

    plot(ts_regrets, ts_stes, df_regrets, df_stes)


if __name__ == '__main__':
    main()