import sys
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import argparse
import pandas as pd
from collections import defaultdict
from argparse import ArgumentParser
from glob import glob
from pathlib import Path
import re
import scipy.stats as stats

pkl_dir = '/Users/sonja/Downloads/dfl/20230107/exps'
BA_WEIGHTS = [0.0, 0.025, 0.05, 0.075, 0.1, 0.125]
PO_WEIGHTS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
DP_WEIGHTS = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

DOMAIN_TO_WEIGHTS = {'toy': DP_WEIGHTS,
                     'babyportfolio': PO_WEIGHTS,
                     'budgetalloc': BA_WEIGHTS,
                    }

BASIC_COLS = ['path', 'mode', 'train_noise', 'test_noise', 'seed', 'test_dq',  'optimal_dq', 'random_dq', 'train_dq', 'val_dq']
LAYER_COLS = ['path', 'mode', 'train_noise', 'test_noise', 'seed', 'layers', 'test_dq',  'optimal_dq', 'random_dq', 'train_dq', 'val_dq']
DIFFICULTY_COLS = ['path', 'mode', 'train_noise', 'test_noise', 'seed', 'layers', 'num_synthetic_layers', 'x_dim', 'faketargets', 'test_dq',  'optimal_dq', 'random_dq', 'train_dq', 'val_dq']

EXP_TYPE_TO_COLS = {'basic': BASIC_COLS,
                    'layer': LAYER_COLS,
                    'difficulty': DIFFICULTY_COLS
                   }

BASIC_REGEX = '_(...)_noise_(.*)_seed_(.*)_add_train_noise_(.)_adversarial_dflalpha_0.0_patience_(.*)_adv_backprop_(.)_test_adversarial_(.*)'
LAYER_REGEX = '_(...)_noise_(.*)_seed_(.*)_add_train_noise_(.)_adversarial_dflalpha_0.0_patience_(.*)_adv_backprop_(.)_test_adversarial_(.*)_layers_(.*)'
DIFFICULTY_REGEX = '_(...)_noise_(.*)_seed_(.*)_add_train_noise_(.)_adversarial_dflalpha_0.0_patience_(.*)_adv_backprop_(.)_test_adversarial_(.*)_layers_(.*)_hidden_dim_(.*)_num_synthetic_layers_(.*)_x_dim_(.*)_faketargets_(.*)'

EXP_TYPE_TO_REGEX = {'basic': BASIC_REGEX,
                    'layer': LAYER_REGEX,
                    'difficulty': DIFFICULTY_REGEX
                   }

DOMAIN_TO_NAME = {'toy': "Demand Prediction",
                  'babyportfolio': "Portfolio Optimization",
                  'budgetalloc': "Budget Allocation",
                 }

NUM_DECIMAL_PLACES = 5


DOMAIN_TO_DEFAULT_X_DIM = {'toy': 10,
                            'babyportfolio': 10,
                            'budgetalloc': 5,
                          }
DOMAIN_TO_X_DIM_OPTIONS = {'toy': [1, 2, 5, 10],
                           'babyportfolio': [2, 5, 10],
                           'budgetalloc': [2, 5, 10],
                 }
FAKE_TARGET_OPTIONS = [0, 5, 10, 20]
# DOMAIN_TO_FAKE_TARGET_OPTIONS = {'toy': [0, 5, 10, 20],
#                                  'babyportfolio': [0, 5, 10, 20],  # FIXME
#                                  'budgetalloc': [0, 5, 10, 20],
#                  }

DOMAIN_TO_MILD_NOISE = {'toy': 1.0,
                        'babyportfolio': 0.1,
                        'budgetalloc': 0.025,
                       }
### PSEUDO

# arguments: 
#   experiment folder
#   domain
#   what data to focus on (full robust data, diagonal robust data, or non-robust data)
#   what types of analyses to produce
#   whether to write graphs
#   whether to calculate *regret* (by subtracting from the optimization)
#   (possibly) whether to enforce that the same seeds (in TS and DFL) are present
#   (possibly) whether to write a csv of non-aggregated data (which combines across all the pickles)
#   (possibly) type of string formatting for the output experiment
#   (possibly) other parameters you want to pull out and conduct analysis on.

def get_weights(domain):
    return DOMAIN_TO_WEIGHTS[domain]

def get_columns(exp_type):
    return EXP_TYPE_TO_COLS[exp_type]

def get_file_regexes(domain, data_cut, exp_type):
    regexes = []
    weights = get_weights(domain)
    if data_cut == 'standard' or data_cut == 'diag_robust':
        regexes.append(f'{domain}_*_noise_0.0_seed_*')
    elif data_cut == 'diag_robust':
        for weight in weights:
            if exp_type == 'basic':
                regexes.append(f'{domain}_*_noise_{weight}_seed_*_test_adversarial_{weight}*')
            else:
                regexes.append(f'{domain}_*_noise_{weight}_seed_*_test_adversarial_{weight}_layers*')

    elif data_cut == 'full_robust':
        # for weight_i in weights:
        #     for weight_j in weights:
        #     regexes.append(f'{domain}_*_noise_{weight_i}_seed_*_test_adversarial_{weight_j}*')
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
    df = pd.DataFrame(raw_rows, columns=columns).drop_duplicates().dropna()  # TODO double check this!
    
    # clean df
    weights = get_weights(domain)
    df = df[(df['train_noise'].isin(weights)) & (df['test_noise'].isin(weights))]
    df.replace('mse', 'ts', inplace=True)
    return df


def make_export_tsv(grouped, save_name):
    for mode in ['ts', 'dfl']:
        mode_df = grouped.loc[grouped['mode'] == mode]
        for stat in ['mean', 'double_std']:  # 2*std... is that the right thing?
            mode_stat = mode_df[['train_noise', 'test_noise', stat]]
            stat_pivot = pd.pivot_table(mode_stat, values=stat, index='train_noise', columns='test_noise').round(NUM_DECIMAL_PLACES)
            stat_pivot.to_csv(save_name + f'_table_{mode}_{stat}', sep='\t')


def double_std(array):
    return np.std(array) * 2


def make_graph(grouped, df, data_cut, domain, save_name):
    dfl = grouped.loc[(grouped['mode'] == 'dfl') & (grouped['train_noise'] == 0)]
    ts = grouped.loc[(grouped['mode'] == 'ts') & (grouped['train_noise'] == 0)]
    optimal = df.groupby(['test_noise']).agg([np.mean, double_std])['optimal_dq'].reset_index()
    random = df.groupby(['test_noise']).agg([np.mean, double_std])['random_dq'].reset_index()
    dfl['DFL'] = dfl['mean']
    ts['TS'] = ts['mean']
    optimal['Optimal'] = optimal['mean']
    random['Random'] = random['mean']

    ax = dfl.plot(x='test_noise', y='DFL', yerr='double_std', ylabel='Decision Quality', xlabel='Test Noise', title=DOMAIN_TO_NAME[domain], color='#bf6c78', capsize=5, zorder=10)
    ts.plot(x='test_noise', y='TS', yerr='double_std', ax=ax,  color='#302583', alpha=0.7, capsize=5)
    optimal.plot(x='test_noise', y='Optimal', yerr='double_std', ax=ax, color='#bdbdbd', alpha=0.8, capsize=5)
    random.plot(x='test_noise', y='Random', yerr='double_std', ax=ax, color='#737373', alpha=0.8, capsize=5)

    if data_cut == 'diag_robust' or data_cut == 'full_robust':
        robust_dfl = grouped.loc[(grouped['mode'] == 'dfl') & (grouped['train_noise'] == grouped['test_noise']) & (grouped['train_noise'] != 0)]
        robust_ts = grouped.loc[(grouped['mode'] == 'ts') & (grouped['train_noise'] == grouped['test_noise']) & (grouped['train_noise'] != 0)]
        robust_dfl['Robust DFL'] = robust_dfl['mean']
        robust_ts['Robust TS'] = robust_ts['mean']
        robust_dfl.plot(x='test_noise', y='Robust DFL', yerr='double_std', ax=ax, color='#4da899', alpha=0.9, capsize=5, zorder=8)
        robust_ts.plot(x='test_noise', y='Robust TS', yerr='double_std', ax=ax, color='#dad067', alpha=1, capsize=5)

    plt.savefig(save_name + '_test_noise_vs_dq.png')
    print('Wrote to ' + save_name + '_test_noise_vs_dq.png')
    

def make_difficulty_graphs(df, domain, save_name):
    df = df[['mode', 'test_noise', 'test_dq', 'num_synthetic_layers', 'x_dim', 'faketargets']]
    default_x_dim = DOMAIN_TO_DEFAULT_X_DIM[domain]
    layer_df = df[(df['x_dim'] == default_x_dim) & (df['faketargets'] == 0)]
    x_dim_df = df[(df['num_synthetic_layers'] == 2) & (df['faketargets'] == 0)]
    fake_targets_df = df[(df['num_synthetic_layers'] == 2) & (df['x_dim'] == default_x_dim)]
    noise_layer_data = {}  # for 0.1
    noise_x_dim_data = {}  # for 0.1
    noise_fake_targets_data = {}  # for 0.1
    mild_noise = DOMAIN_TO_MILD_NOISE[domain]

    fig, ax = plt.subplots()

    # num_synthetic_layers
    for mode in ['dfl', 'ts']:
        mode_layer_df = layer_df.loc[layer_df['mode'] == mode]
        mode_layer_grouped = mode_layer_df[['test_noise', 'num_synthetic_layers', 'test_dq']].groupby(['test_noise', 'num_synthetic_layers']).agg([np.mean, double_std])['test_dq'].reset_index()
        noise_layer_data[mode] = mode_layer_grouped.loc[mode_layer_grouped['test_noise'] == mild_noise]
        # for num_layers in range(1, 6):
        for num_layers in range(1, 6):
            curr_mode_layer = mode_layer_grouped.loc[mode_layer_grouped['num_synthetic_layers'] == num_layers]
            curr_mode_layer[f'{mode}_{num_layers}'] = curr_mode_layer['mean']
            curr_mode_layer.plot(x='test_noise', y=f'{mode}_{num_layers}', yerr='double_std', ax=ax, linewidth=1, capsize=5, title=f'Number of Synthetic Layers: {DOMAIN_TO_NAME[domain]}',  ylabel='Decision Quality', xlabel='Test Noise')
            # curr_mode_layer.plot(x='test_noise', y=f'{mode}_{num_layers}', ax=ax, linewidth=1)

    plt.savefig(save_name + '_difficulty_num_synthetic_layers.png')
    print('Wrote to ' + save_name + '_difficulty_num_synthetic_layers.png')
    plt.cla()

    noise_layer_data['dfl']['DFL'] = noise_layer_data['dfl']['mean']
    noise_layer_data['ts']['TS'] = noise_layer_data['ts']['mean']
    noise_layer_data['dfl'].plot(x='num_synthetic_layers', y='DFL', yerr='double_std', ax=ax, linewidth=1, capsize=5, title=f'Number of Synthetic Layers: {DOMAIN_TO_NAME[domain]}',  ylabel='Decision Quality', xlabel='Number of Synthetic Layers')
    noise_layer_data['ts'].plot(x='num_synthetic_layers', y='TS', yerr='double_std', ax=ax, linewidth=1, capsize=5, title=f'Number of Synthetic Layers: {DOMAIN_TO_NAME[domain]}',  ylabel='Decision Quality', xlabel='Number of Synthetic Layers')
    
    plt.savefig(save_name + f'_difficulty_num_synthetic_layers_{mild_noise}.png')
    print('Wrote to ' + save_name + f'_difficulty_num_synthetic_layers_{mild_noise}.png')
    plt.cla()    

    # x_dim
    for mode in ['dfl', 'ts']:
        mode_x_dim_df = x_dim_df.loc[x_dim_df['mode'] == mode]
        mode_x_dim_grouped = mode_x_dim_df[['test_noise', 'x_dim', 'test_dq']].groupby(['test_noise', 'x_dim']).agg([np.mean, double_std])['test_dq'].reset_index()
        noise_x_dim_data[mode] = mode_x_dim_grouped.loc[mode_x_dim_grouped['test_noise'] == mild_noise]
        # for num_layers in range(1, 6):
        for x_dim in DOMAIN_TO_X_DIM_OPTIONS[domain]:
            curr_mode_x_dim = mode_x_dim_grouped.loc[mode_x_dim_grouped['x_dim'] == x_dim]
            curr_mode_x_dim[f'{mode}_{x_dim}'] = curr_mode_x_dim['mean']
            curr_mode_x_dim.plot(x='test_noise', y=f'{mode}_{x_dim}', yerr='double_std', ax=ax, linewidth=1, capsize=5, title=f'Number of X Dimensions: {DOMAIN_TO_NAME[domain]}', ylabel='Decision Quality', xlabel='Test Noise')
            # curr_mode_layer.plot(x='test_noise', y=f'{mode}_{num_layers}', ax=ax, linewidth=1)
    
    plt.savefig(save_name + '_difficulty_x_dim.png')
    print('Wrote to ' + save_name + '_difficulty_x_dim.png')
    plt.cla()

    noise_x_dim_data['dfl']['DFL'] = noise_x_dim_data['dfl']['mean']
    noise_x_dim_data['ts']['TS'] = noise_x_dim_data['ts']['mean']
    noise_x_dim_data['dfl'].plot(x='x_dim', y='DFL', yerr='double_std', ax=ax, linewidth=1, capsize=5, title=f'X Dimension: {DOMAIN_TO_NAME[domain]}',  ylabel='Decision Quality', xlabel='X Dimension')
    noise_x_dim_data['ts'].plot(x='x_dim', y='TS', yerr='double_std', ax=ax, linewidth=1, capsize=5, title=f'X Dimension: {DOMAIN_TO_NAME[domain]}',  ylabel='Decision Quality', xlabel='X Dimension')
    
    plt.savefig(save_name + f'_difficulty_x_dim_{mild_noise}.png')
    print('Wrote to ' + save_name + f'_difficulty_x_dim_{mild_noise}.png')
    plt.cla() 

    # faketargets
    for mode in ['dfl', 'ts']:
        mode_fake_targets_df = fake_targets_df.loc[fake_targets_df['mode'] == mode]
        mode_fake_targets_grouped = mode_fake_targets_df[['test_noise', 'faketargets', 'test_dq']].groupby(['test_noise', 'faketargets']).agg([np.mean, double_std])['test_dq'].reset_index()
        noise_fake_targets_data[mode] = mode_fake_targets_grouped.loc[mode_fake_targets_grouped['test_noise'] == mild_noise]
        # for num_layers in range(1, 6):
        for num_fake_targets in FAKE_TARGET_OPTIONS:
            curr_mode_fake_targets = mode_fake_targets_grouped.loc[mode_fake_targets_grouped['faketargets'] == num_fake_targets]
            curr_mode_fake_targets[f'{mode}_{num_fake_targets}'] = curr_mode_fake_targets['mean']
            curr_mode_fake_targets.plot(x='test_noise', y=f'{mode}_{num_fake_targets}', yerr='double_std', ax=ax, linewidth=1, capsize=5, title=f'Number of Fake Targets: {DOMAIN_TO_NAME[domain]}',  ylabel='Decision Quality', xlabel='Test Noise')
            # curr_mode_layer.plot(x='test_noise', y=f'{mode}_{num_layers}', ax=ax, linewidth=1)
    
    plt.savefig(save_name + '_difficulty_fake_targets.png')
    print('Wrote to ' + save_name + '_difficulty_fake_targets.png')
    plt.cla()

    noise_fake_targets_data['dfl']['DFL'] = noise_fake_targets_data['dfl']['mean']
    noise_fake_targets_data['ts']['TS'] = noise_fake_targets_data['ts']['mean']
    noise_fake_targets_data['dfl'].plot(x='faketargets', y='DFL', yerr='double_std', ax=ax, linewidth=1, capsize=5, title=f'Number of Fake Targets: {DOMAIN_TO_NAME[domain]}',  ylabel='Decision Quality', xlabel='Number of Fake Targets')
    noise_fake_targets_data['ts'].plot(x='faketargets', y='TS', yerr='double_std', ax=ax, linewidth=1, capsize=5, title=f'Number of Fake Targets: {DOMAIN_TO_NAME[domain]}',  ylabel='Decision Quality', xlabel='Number of Fake Targets')
    
    plt.savefig(save_name + f'_difficulty_fake_targets_{mild_noise}.png')
    print('Wrote to ' + save_name + f'_difficulty_fake_targets_{mild_noise}.png')
    plt.cla() 

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
    grouped = df.groupby(['mode','train_noise', 'test_noise']).agg([np.mean, double_std])['test_dq'].reset_index()
    save_name = str(Path(args.out_dir) / f'{args.out_name}_{domain}_{data_cut}_{exp_type}')
    # make export tsv
    make_export_tsv(grouped, save_name)
    # graph
    make_graph(grouped, df, data_cut, domain, save_name)

    if exp_type == 'difficulty':
        make_difficulty_graphs(df, domain, save_name)







if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str)
    parser.add_argument('--problem', type=str, choices=['budgetalloc', 'toy', 'babyportfolio'])
    parser.add_argument('--data_cut', type=str, choices=['full_robust', 'diag_robust', 'standard'])
    parser.add_argument('--exp_type', type=str, choices=['basic', 'layer', 'difficulty'])
    parser.add_argument('--analysis', type=str, choices=['mean', 'scatter'])  # perhaps the option is to add scatters on top
    parser.add_argument('--out_dir', type=str, default='./')
    parser.add_argument('--out_name', type=str, required=True)
    # parser.add_argument('--fig', type=ast.literal_eval, default=True)
    # parser.add_argument('--as_regret', type=ast.literal_eval, default=False)
    # parser.add_argument('--write_disagg', type=ast.literal_eval, default=False)

    args = parser.parse_args()

    analyze(args)













# def print_results(add_train_noise=0, adv_backprop=0):
#     ts_ba = pd.DataFrame({})
#     df_ba = pd.DataFrame({})
#     test_noise_level = 0.2
#     layers = 1
#     print('just doing 10')
#     ts_rewards = defaultdict(list)
#     df_rewards = defaultdict(list)
#     for mode in ['mse', 'dfl']:
#         for seed in range(10):
#             all_present = True
#             data_to_add_mse = {}
#             data_to_add_dfl = {}
#             for noise_level in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        
# #                 test_noise_level = noise_level
#                 try:
#                     with open(f'{pkl_dir}/babyportfolio_{mode}_noise_{noise_level}_seed_{seed}_add_train_noise_{add_train_noise}_adversarial_dflalpha_0.0_patience_100_adv_backprop_{adv_backprop}_test_adversarial_{test_noise_level}_layers_{layers}', 'rb') as f:
#                         data = pickle.load(f)
#                     if mode == 'mse':
# #                         print(data['optimal'])
#                         data_to_add_mse[noise_level] = data['test'] #- data['random']
#                     else:
#                         data_to_add_dfl[noise_level] = data['test'] #- data['random']
#                 except:
#                     all_present = False
#                     if not (add_train_noise == 0 and adv_backprop == 1):
#                         print(f'{pkl_dir}/babyportfolio_{mode}_noise_{noise_level}_seed_{seed}_add_train_noise_{add_train_noise}_adversarial_dflalpha_{ts_weight}_patience_100_adv_backprop_{adv_backprop}_test_adversarial_{test_noise_level}_layers_{layers} does not exist')
#             if all_present:
#                 for noise_level in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
# #                 for noise_level in [0.0, 0.5, 1.0, 3.0, 5.0, 10.0, 20.0, 100.0]:
#                     if mode == 'mse':
#                         ts_rewards[noise_level].append(data_to_add_mse[noise_level])
#                     else:
#                         df_rewards[noise_level].append(data_to_add_dfl[noise_level])

#     ts_xs = []
#     ts_ys = []
#     df_xs = []
#     df_ys = []
#     print(data['optimal'])
#     print()
#     for noise_level in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
# #     for noise_level in [0.0, 0.5, 1.0, 3.0, 5.0, 10.0, 20.0, 100.0]:
#         ts_ba[noise_level] = ts_rewards[noise_level]
#         df_ba[noise_level] = df_rewards[noise_level]
#         ts_xs += [noise_level] * len(ts_rewards[noise_level])
#         df_xs += [noise_level] * len(df_rewards[noise_level])
#         ts_ys += list(ts_rewards[noise_level])
#         df_ys += list(df_rewards[noise_level])

#     plt.scatter(ts_xs, ts_ys, alpha=0.3, label='ts')
#     plt.scatter(df_xs, df_ys, alpha=0.3, label='dfl')
#     plt.plot(np.unique(ts_xs), np.poly1d(np.polyfit(ts_xs, ts_ys, 1))(np.unique(ts_xs)), label='ts')
#     plt.plot(np.unique(df_xs), np.poly1d(np.polyfit(df_xs, df_ys, 1))(np.unique(df_xs)), label='dfl')

#     plt.title('Portfolio Optimization: Robust DFL vs Robust TS')
#     plt.ylabel('Decision Quality')
#     plt.xlabel('Adversarial Noise Budget')
#     plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
# #     print('ts')
# #     print_arr(ts_ba)
# #     print(ts_ba)
#     print_arr(ts_ba.mean().values)

#     ts_ba.to_csv('po_robust_ts_test.csv')
# #     print_arr(ts_ba.describe())

#     print('df')
# #     print(df_ba)
#     df_ba.to_csv('po_robust_df_test.csv')
#     print_arr(df_ba.mean().values)
# #     print_arr(df_ba.describe())

# #     return ts_ba, df_ba

# def print_arr(arr):
#     for elem in arr:
#         print(elem)

# for ts_weight in [0.0]:
#     for add_train_noise in [0, 1]:
#         for adv_backprop in [0]:
#             print(ts_weight, 'adversarial' if add_train_noise == 0 else 'adversarial_training', 'adv_backprop' if adv_backprop else '')
#             print_results(ts_weight, add_train_noise=add_train_noise, adv_backprop=adv_backprop)
#             print()
#         print()