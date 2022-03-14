import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--path', default='.', type=str, help='string name')

    args = parser.parse_args()
    prefixes = ['TS', 'DF_IS', 'DF_SIM']
    filepaths = [Path('results') / (prefix + '_' + args.path + '_sd_0.pickle') for prefix in prefixes]
    data = {}
    reward_data = {}

    for i, filepath in enumerate(filepaths):
        with open(filepath, 'rb') as f:
            data[prefixes[i]] = pickle.load(f)
            reward_data[prefixes[i]] = {}
            reward_data[prefixes[i]]['DF_IS'] = np.array(data[prefixes[i]][1]['test'])
            reward_data[prefixes[i]]['DF_SIM'] = np.array(data[prefixes[i]][2]['test'])
            reward_data[prefixes[i]]['DF_SIM_optimal'] = np.array(data[prefixes[i]][-1]['test'])
            reward_data[prefixes[i]]['DF_IS_optimal'] = np.array(data[prefixes[i]][-2]['test'])
 
    # TODO: run with smaller noise
    # TODO better: take the largest reward in the val set and then select the epoch in the test set
    # TODO: use sim OPE, not IS OPE
    # NOTE: we don't totally trust DF-sim
    # TODO: remove this
    prefixes = ['TS', 'DF_SIM']

    df_is_ope_graph = [np.max(reward_data[prefix]['DF_IS']) for prefix in prefixes]
    df_sim_ope_graph = [np.max(reward_data[prefix]['DF_SIM']) for prefix in prefixes]
    df_is_ope_optimal_graph = [np.max(reward_data[prefix]['DF_IS_optimal']) for prefix in prefixes]
    df_sim_ope_optimal_graph = [np.max(reward_data[prefix]['DF_SIM_optimal']) for prefix in prefixes]


    # # plt.bar(prefixes + ['opt DF_IS', 'opt DF_sim'], df_is_ope_graph + [np.mean(df_is_ope_optimal_graph), np.mean(df_sim_ope_optimal_graph)])
    # plt.bar(prefixes + ['opt DF_sim'], df_is_ope_graph + [np.mean(df_sim_ope_optimal_graph)])
    # plt.title('Reward from ISO OPE')
    # plt.ylabel('Reward')
    # # plt.show()
    # plt.savefig(args.path + '_reward_iso_ope.png')
    # plt.show()

    # plt.bar(prefixes + ['opt DF_IS', 'opt DF_sim'], df_sim_ope_graph + [np.mean(df_is_ope_optimal_graph), np.mean(df_sim_ope_optimal_graph)])
    plt.bar(prefixes + ['opt DF_sim'], df_sim_ope_graph + [np.mean(df_sim_ope_optimal_graph)])

    plt.title('Reward from Sim OPE')
    plt.ylabel('Reward')
    # plt.show()
    plt.savefig(args.path + '_reward_sim_ope.png')
    # regret_data = {}

    # for i, filepath in enumerate(filepaths):
    #     with open(filepath, 'rb') as f:
    #         data[prefixes[i]] = pickle.load(f)
    #         regret_data[prefixes[i]] = {}
    #         regret_data[prefixes[i]]['DF_IS'] = np.array(data[prefixes[i]][-2]['test']) - np.array(data[prefixes[i]][1]['test'])
    #         regret_data[prefixes[i]]['DF_SIM'] = np.array(data[prefixes[i]][-1]['test']) - np.array(data[prefixes[i]][2]['test'])
 
    # df_is_ope_graph = [np.mean(regret_data[prefix]['DF_IS']) for prefix in prefixes]
    # df_sim_ope_graph = [np.mean(regret_data[prefix]['DF_SIM']) for prefix in prefixes]

    # plt.bar(prefixes, df_is_ope_graph)
    # plt.title('Regret from ISO OPE')
    # plt.ylabel('Regret')
    # # plt.show()
    # plt.savefig(args.path + '_iso_ope.png')
    # plt.show()

    # plt.bar(prefixes, df_sim_ope_graph)
    # plt.title('Regret from Sim OPE')
    # plt.ylabel('Regret')
    # # plt.show()
    # plt.savefig(args.path + '_sim_ope.png')
