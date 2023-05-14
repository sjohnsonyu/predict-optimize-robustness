import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.pyplot as mpl
from matplotlib.ticker import ScalarFormatter,AutoMinorLocator
from constants import *
# import scienceplots

# plt.style.use('science')

dfl_color = '#AD1459'
ts_color = '#2859AD'
opt_color = '#737373'
rdfl_color = '#60A41E'
rts_color = '#FFB907'
fontname = {'fontname':'Times'}


def get_err(err_arr, mean_arr):
    bounds = []
    for err, mean in zip(err_arr, mean_arr):
        err = (err / 2) / np.sqrt(30)  # calculated double std
        lower = err
        upper = err
        if mean + upper > 1:
            upper = 1 - mean
        bounds.append([lower, upper])
    return np.array(bounds).T

def make_graph(grouped, df, data_cut, domain, save_name):
    test_noise = 'Adversarial Noise Budget'
    grouped = grouped.rename(columns={'test_noise': test_noise})
    df = df.rename(columns={'test_noise': test_noise})
    dfl = grouped.loc[(grouped['mode'] == 'dfl') & (grouped['train_noise'] == 0)]
    ts = grouped.loc[(grouped['mode'] == 'ts') & (grouped['train_noise'] == 0)]
    optimal = df.groupby([test_noise]).agg([np.mean, double_std])['optimal_dq'].reset_index()
    random = df.groupby([test_noise]).agg([np.mean, double_std])['random_dq'].reset_index()
    dfl['DFL'] = dfl['mean']
    ts['TS'] = ts['mean']
    optimal['Optimal'] = optimal['mean']
    random['Random'] = random['mean']
    rc('font',**{'family':'serif','serif':['Times']})
    mpl.rcParams['font.size'] = 16
    plt.tight_layout()
    # ax = dfl.plot(x='test_noise', y='DFL', yerr='double_std', ylabel='Decision Quality', xlabel='Test Noise', title=DOMAIN_TO_NAME[domain], color=dfl_color, capsize=5, zorder=10)
    dfl_err = get_err(dfl['double_std'],  dfl['DFL'])
    ts_err = get_err(ts['double_std'],  ts['TS'])
    ax = dfl.plot(x=test_noise, y='DFL', yerr=dfl_err, ylabel='Decision Quality', xlabel='Test Noise', color=dfl_color, capsize=5, zorder=10, figsize=(6, 2.5), legend=False)
    # ax = dfl.plot(x=test_noise, y='DFL', yerr=dfl_err, ylabel='Decision Quality', xlabel='Test Noise', color=dfl_color, capsize=5, zorder=10, figsize=(6, 4))
    # ax = format_ax(ax)
    ts.plot(x=test_noise, y='TS', yerr=ts_err, ax=ax,  color=ts_color, alpha=0.7, capsize=5, legend=False)
    
    # random.plot(x=test_noise, y='Random', yerr='double_std', ax=ax, color='#737373', alpha=0.8, capsize=5)
    

    if data_cut == 'diag_robust' or data_cut == 'full_robust':
        robust_dfl = grouped.loc[(grouped['mode'] == 'dfl') & (grouped['train_noise'] == grouped[test_noise]) & (grouped['train_noise'] != 0)]
        robust_ts = grouped.loc[(grouped['mode'] == 'ts') & (grouped['train_noise'] == grouped[test_noise]) & (grouped['train_noise'] != 0)]
        robust_dfl['Robust DFL'] = robust_dfl['mean']
        robust_ts['Robust TS'] = robust_ts['mean']
        # rdfl_err = [(max(0, err), min(1, err)) for err in robust_dfl['double_std']]
        rdfl_err = get_err(robust_dfl['double_std'],  robust_dfl['Robust DFL'])
        rts_err = get_err(robust_ts['double_std'],  robust_ts['Robust TS'])

        robust_dfl.plot(x=test_noise, y='Robust DFL', yerr=rdfl_err, ax=ax, color=rdfl_color, alpha=0.9, capsize=5, zorder=8, legend=False)
        robust_ts.plot(x=test_noise, y='Robust TS', yerr=rts_err, ax=ax, color=rts_color, alpha=1, capsize=5, legend=False)

        

    # optimal.plot(x=test_noise, y='Optimal', yerr='double_std', ax=ax, color=opt_color, alpha=0.8, capsize=5)   
    # ax.legend(ncol=4, bbox_to_anchor=(-0.1, 1, 0.5, 0), columnspacing=0.5, fontsize=14)
    # ax.legend(ncol=1, bbox_to_anchor=(0.9, 1.05, 0.5, 0), columnspacing=0.5, fontsize=14)
    plt.savefig(save_name + '_test_noise_vs_dq.png', bbox_inches='tight')
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

    _, ax = plt.subplots()
    plt.gcf().subplots_adjust(bottom=0.15)

    # num_synthetic_layers
    for mode in ['dfl', 'ts']:
        mode_layer_df = layer_df.loc[layer_df['mode'] == mode]
        mode_layer_grouped = mode_layer_df[['test_noise', 'num_synthetic_layers', 'test_dq']].groupby(['test_noise', 'num_synthetic_layers']).agg([np.mean, double_std])['test_dq'].reset_index()
        noise_layer_data[mode] = mode_layer_grouped.loc[mode_layer_grouped['test_noise'] == mild_noise]
        for num_layers in range(1, 6):
            curr_mode_layer = mode_layer_grouped.loc[mode_layer_grouped['num_synthetic_layers'] == num_layers]
            curr_mode_layer[f'{mode}_{num_layers}'] = curr_mode_layer['mean']
            curr_mode_layer.plot(x='test_noise', y=f'{mode}_{num_layers}', yerr='double_std', ax=ax, linewidth=1, capsize=5, title=f'Number of Synthetic Layers: {DOMAIN_TO_NAME[domain]}',  ylabel='Decision Quality', xlabel='Test Noise')

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
        for x_dim in DOMAIN_TO_X_DIM_OPTIONS[domain]:
            curr_mode_x_dim = mode_x_dim_grouped.loc[mode_x_dim_grouped['x_dim'] == x_dim]
            curr_mode_x_dim[f'{mode}_{x_dim}'] = curr_mode_x_dim['mean']
            curr_mode_x_dim.plot(x='test_noise', y=f'{mode}_{x_dim}', yerr='double_std', ax=ax, linewidth=1, capsize=5, title=f'Number of X Dimensions: {DOMAIN_TO_NAME[domain]}', ylabel='Decision Quality', xlabel='Test Noise')
    
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
        for num_fake_targets in FAKE_TARGET_OPTIONS:
            curr_mode_fake_targets = mode_fake_targets_grouped.loc[mode_fake_targets_grouped['faketargets'] == num_fake_targets]
            curr_mode_fake_targets[f'{mode}_{num_fake_targets}'] = curr_mode_fake_targets['mean']
            curr_mode_fake_targets.plot(x='test_noise', y=f'{mode}_{num_fake_targets}', yerr='double_std', ax=ax, linewidth=1, capsize=5, title=f'Number of Fake Targets: {DOMAIN_TO_NAME[domain]}',  ylabel='Decision Quality', xlabel='Test Noise')
    
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


def make_layer_graphs(df, domain, save_name):
    layer_df = df[['mode', 'test_noise', 'test_dq', 'layers']]
    noise_layer_data = {}  # for 0.1
    mild_noise = DOMAIN_TO_MILD_NOISE[domain]

    _, ax = plt.subplots()

    # num_synthetic_layers
    for mode in ['dfl', 'ts']:
        mode_layer_df = layer_df.loc[layer_df['mode'] == mode]
        mode_layer_grouped = mode_layer_df[['test_noise', 'layers', 'test_dq']].groupby(['test_noise', 'layers']).agg([np.mean, double_std])['test_dq'].reset_index()
        noise_layer_data[mode] = mode_layer_grouped.loc[mode_layer_grouped['test_noise'] == mild_noise]
        for num_layers in range(1, 6):
            curr_mode_layer = mode_layer_grouped.loc[mode_layer_grouped['layers'] == num_layers]
            curr_mode_layer[f'{mode}_{num_layers}'] = curr_mode_layer['mean']
            curr_mode_layer.plot(x='test_noise', y=f'{mode}_{num_layers}', yerr='double_std', ax=ax, linewidth=1, capsize=5, title=f'Number of Layers: {DOMAIN_TO_NAME[domain]}',  ylabel='Decision Quality', xlabel='Test Noise')

    plt.savefig(save_name + '_layers.png')
    print('Wrote to ' + save_name + '_layers.png')
    plt.cla()

    noise_layer_data['dfl']['DFL'] = noise_layer_data['dfl']['mean']
    noise_layer_data['ts']['TS'] = noise_layer_data['ts']['mean']
    noise_layer_data['dfl'].plot(x='layers', y='DFL', yerr='double_std', ax=ax, linewidth=1, capsize=5, title=f'Number of Layers: {DOMAIN_TO_NAME[domain]}',  ylabel='Decision Quality', xlabel='Number of Layers')
    noise_layer_data['ts'].plot(x='layers', y='TS', yerr='double_std', ax=ax, linewidth=1, capsize=5, title=f'Number of Layers: {DOMAIN_TO_NAME[domain]}',  ylabel='Decision Quality', xlabel='Number of Layers')
    
    plt.savefig(save_name + f'_layers_{mild_noise}.png')
    print('Wrote to ' + save_name + f'_layers_{mild_noise}.png')
    plt.cla()

def format_ax(ax):
    ax.set_xlabel('')
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.major.formatter._useMathText = True
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    # ax.yaxis.set_label_coords(0.63,1.01)
    # ax.yaxis.tick_right()
    # ax.legend(frameon=False, loc='upper right',ncol=2,handlelength=4)
    return ax


# Move to own utils file 
def double_std(array):
    return np.std(array) * 2
