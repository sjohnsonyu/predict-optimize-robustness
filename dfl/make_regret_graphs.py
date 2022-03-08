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
    regret_data = {}

    for i, filepath in enumerate(filepaths):
        with open(filepath, 'rb') as f:
            data[prefixes[i]] = pickle.load(f)
            regret_data[prefixes[i]] = {}
            regret_data[prefixes[i]]['DF_IS'] = data[prefixes[i]][-2]['test']
            regret_data[prefixes[i]]['DF_SIM'] = data[prefixes[i]][-1]['test']
 
    df_is_ope_graph = [np.mean(regret_data[prefix]['DF_IS']) for prefix in prefixes]
    df_sim_ope_graph = [np.mean(regret_data[prefix]['DF_SIM']) for prefix in prefixes]

    plt.bar(prefixes, df_is_ope_graph)
    plt.title('Regret from ISO OPE')
    plt.ylabel('Regret')
    # plt.show()
    plt.savefig(args.path + '_iso_ope.png')
    plt.show()

    plt.bar(prefixes, df_sim_ope_graph)
    plt.title('Regret from Sim OPE')
    plt.ylabel('Regret')
    # plt.show()
    plt.savefig(args.path + '_sim_ope.png')
