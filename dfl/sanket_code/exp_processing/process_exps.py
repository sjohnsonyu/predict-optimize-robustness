import numpy as np
import pickle
import pandas as pd
from argparse import ArgumentParser
from glob import glob
from pathlib import Path
import re
from graphing_utils import make_graph, make_difficulty_graphs, double_std, make_layer_graphs
from constants import *


def get_weights(domain):
    return DOMAIN_TO_WEIGHTS[domain]


def get_columns(exp_type):
    return EXP_TYPE_TO_COLS[exp_type]


def get_file_regexes(domain, data_cut, exp_type):
    regexes = []
    weights = get_weights(domain)
    if data_cut == 'standard' or data_cut == 'diag_robust':
        regexes.append(f'{domain}_*_noise_0.0_seed_*')
    if data_cut == 'diag_robust':
        for weight in weights:
            if exp_type == 'basic':
                regexes.append(f'{domain}_*_noise_{weight}_seed_*_test_adversarial_{weight}*')
            else:
                regexes.append(f'{domain}_*_noise_{weight}_seed_*_test_adversarial_{weight}_layers*')

    elif data_cut == 'full_robust':
        regexes.append(f'{domain}_*')
    return regexes


def get_filenames(dir_path, domain, data_cut, exp_type):
    dir_path = Path(dir_path)
    filenames = []
    file_regexes = get_file_regexes(domain, data_cut, exp_type)
    for file_regex in file_regexes:
        filenames += glob(str(dir_path / file_regex))
    assert len(filenames) > 0
    return filenames


def get_row(filename, exp_type):
    row = []
    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        params = re.split(EXP_TYPE_TO_REGEX[exp_type], filename)
        mode = params[1]
        train_noise = float(params[2])
        seed = int(params[3])
        add_train_noise = int(params[4])  # not currently used
        patience = int(params[5])  # not currently used
        adv_backprop = int(params[6])  # not currently used
        test_noise = float(params[7])

        if exp_type != 'basic':
            layers = int(params[8])
        if exp_type == 'difficulty':
            hidden_dim = int(params[9])  # not currently used
            num_synthetic_layers = int(params[10])
            x_dim = int(params[11])
            faketargets = int(params[12])

        test_dq = data['test']
        random_dq = data['random']
        optimal_dq = data['optimal']
        train_dq = data['train']
        val_dq = data['val']
        train_loss = data['train_loss']
        val_loss = data['val_loss']
        test_loss = data['test_loss']

        row = [filename, mode, train_noise, test_noise, seed]
        if exp_type != 'basic':
            row.append(layers)
        if exp_type == 'difficulty':
            row += [num_synthetic_layers, x_dim, faketargets]

        row += [test_dq,  optimal_dq, random_dq, train_dq, val_dq]
    except:
        print(f'Skipping {filename}')
    return row
    

def create_df(filenames, exp_type, domain):
    raw_rows = []
    for filename in filenames:
        raw_row = get_row(filename, exp_type)
        raw_rows.append(raw_row)
    columns = get_columns(exp_type)
    df = pd.DataFrame(raw_rows, columns=columns).drop_duplicates().dropna()
    
    # clean df
    weights = get_weights(domain)
    df = df[(df['train_noise'].isin(weights)) & (df['test_noise'].isin(weights))]
    df.replace('mse', 'ts', inplace=True)
    return df


def make_export_tsv(grouped, save_name, is_epsilon=False):
    for mode in ['ts', 'dfl']:
        mode_df = grouped.loc[grouped['mode'] == mode]
        for stat in ['mean', 'double_std']:  # 2*std... is that the right thing?
            mode_stat = mode_df[['train_noise', 'test_noise', stat]]
            stat_pivot = pd.pivot_table(mode_stat, values=stat, index='train_noise', columns='test_noise').round(NUM_DECIMAL_PLACES)
            if is_epsilon:
                exp_name = save_name + '_epsilon' + f'_table_{mode}_{stat}'
            else:
                exp_name = save_name + f'_table_{mode}_{stat}'
            stat_pivot.to_csv(exp_name, sep='\t')


def calculate_epsilons(df, save_name):
    df['dq_error'] = df['optimal_dq'] - df['test_dq']
    grouped = df.groupby(['mode','train_noise', 'test_noise']).agg([np.mean, double_std])['dq_error'].reset_index()
    make_export_tsv(grouped, save_name, is_epsilon=True)


def analyze(args):
    dir_path = args.dir
    domain = args.problem
    data_cut = args.data_cut
    exp_type = args.exp_type
    filenames = get_filenames(dir_path=dir_path,
                              domain=domain,
                              data_cut=data_cut,
                              exp_type=exp_type
    )
    df = create_df(filenames, exp_type, domain)
    save_name = str(Path(args.out_dir) / f'{args.out_name}_{domain}_{data_cut}_{exp_type}')
    calculate_epsilons(df, save_name)
    grouped = df.groupby(['mode','train_noise', 'test_noise']).agg([np.mean, double_std])['test_dq'].reset_index()
    # make export tsv
    make_export_tsv(grouped, save_name)
    # graph
    make_graph(grouped, df, data_cut, domain, save_name)

    if args.analysis == 'difficulty':  #
        make_difficulty_graphs(df, domain, save_name)
    elif args.analysis == 'layer':
        make_layer_graphs(df, domain, save_name)

    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dir', type=str)
    parser.add_argument('--problem', type=str, choices=['budgetalloc', 'toy', 'babyportfolio', 'toymod'])
    parser.add_argument('--data_cut', type=str, choices=['full_robust', 'diag_robust', 'standard'])
    parser.add_argument('--exp_type', type=str, choices=['basic', 'layer', 'difficulty'])
    parser.add_argument('--analysis', type=str, choices=['difficulty', 'layer'])  # perhaps the option is to add scatters on top
    parser.add_argument('--out_dir', type=str, default='./')
    parser.add_argument('--out_name', type=str, required=True)
    # parser.add_argument('--fig', type=ast.literal_eval, default=True)
    # parser.add_argument('--as_regret', type=ast.literal_eval, default=False)
    # parser.add_argument('--write_disagg', type=ast.literal_eval, default=False)

    args = parser.parse_args()
    analyze(args)
