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

EPSILONS = np.linspace(0.01, 0.12, 12)
SEEDS = np.arange(0, 40)


def get_success_rate(eps):
    if eps == 0.01:
        return 6/40
    # elif eps == 

    with open(f'eps_summary_{eps}.json', 'w') as f:
        data = json.load(f)
    successes = data[0][eps]
    attempts = data[1][eps]
    return successes/attempts
    # data = json.dump([successes, attempts], f)


def get_avg_df_regret(eps):
    df_sim_outputs = []
    save_name = f'exp_eps_{eps}'
    for seed in SEEDS:
        df_sim_filename = f'./results/DF_SIM_{save_name}_sd_{seed}.pickle'  # TODO update save_name and dir
        try:
            with open (df_sim_filename, 'rb') as df_sim_file:
                loaded_df_sim_file = pickle.load(df_sim_file)
                df_sim_outputs.append(loaded_df_sim_file)
        except:
            pass

    df_sim_selected_metrics = []
    df_regrets = []
    # optimal_selected_metrics = []
    # optimal_opes = []
    for i in range(len(df_sim_outputs)):
        optimal_selected_epoch = np.argmax(df_sim_outputs[i][2]['val'][:-1]) # Maximize SIM OPE
        optimal_performance = df_sim_outputs[i][2]['test'][optimal_selected_epoch]
        # optimal_selected_metrics.append([0, df_sim_outputs[i][2]['test'][optimal_selected_epoch]])
        # optimal_opes.append(optimal_performance)
        df_sim_selected_epoch = np.argmax(df_sim_outputs[i][1]['val'][:-1]) # Maximize SIM OPE
        # df_sim_selected_metrics.append([df_sim_outputs[i][j]['test'][df_sim_selected_epoch] for j in range(2)])
        df_regrets.append(optimal_performance - df_sim_outputs[i][1]['test'][df_sim_selected_epoch])

    df_regret_mean, df_regret_ste = np.mean(df_regrets, axis=0), np.std(df_regrets, axis=0) / np.sqrt(len(df_sim_outputs))
    return df_regret_mean, df_regret_ste


def plot(success_rates, avg_df_regrets, avg_df_stes):
    fig, ax1 = plt.subplots()
    color1 = 'tab:blue'
    ax1.set_xlabel('Epsilon')
    ax1.set_title('Epsilon vs. Success Rate and Regret')
    ax1.set_ylabel('Success Rate', color=color1)
    ax1.plot(EPSILONS, success_rates, color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()

    color2 = 'tab:red'
    ax2.set_ylabel('Avg DF Regret', color=color2)
    ax2.plot(EPSILONS, avg_df_regrets, color=color2)
    ax2.fill_between(EPSILONS, avg_df_regrets-avg_df_stes, avg_df_regrets+avg_df_stes, color=color2, alpha=0.2)  # df loss per epoch
    ax2.tick_params(axis='y', labelcolor=color2)

    fig.tight_layout()
    plt.save_fig('eps_success_regret.png')
    plt.show()


def main():
    success_rates = []
    avg_df_regrets = []
    avg_df_stes = []

    for eps in EPSILONS:
        success_rate = get_success_rate(eps)
        avg_df_regret, avg_df_ste = get_avg_df_regret(eps)
        success_rates.append(success_rate)
        avg_df_regrets.append(avg_df_regret)
        avg_df_stes.append(avg_df_ste)

    success_rates = np.array(success_rates)
    avg_df_regrets = np.array(avg_df_regrets)
    avg_df_stes = np.array(avg_df_stes)

    plot(success_rates, avg_df_regrets, avg_df_stes)


if __name__ == '__main__':
    main()