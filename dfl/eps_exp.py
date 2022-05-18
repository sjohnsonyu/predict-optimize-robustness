import subprocess
import sys
import os
import numpy as np
import itertools
import pickle
import matplotlib.pyplot as plt
import argparse
from train import main as train_main
from copy import copy
from collections import defaultdict
import json

EPSILONS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15]

parser=argparse.ArgumentParser(description = 'Visulation and tuning DF vs TS comparison')
parser.add_argument('--epochs', default=10, type=int, help='Number of epochs')
parser.add_argument('--instances', default=10, type=int, help='Number of instances')
parser.add_argument('--seed', default=0, type=int, help='Random seed')
parser.add_argument('--save', default=0, type=int, help='Whether or not to save all generated figs. Put 0 for False, 1 for True')
parser.add_argument('--plot', default=0, type=int, help='Whether or not to create plots. Put 0 for False, 1 for True')
parser.add_argument('--compute', default=1, type=int, help='Whether or not to run new experiments. Put 0 for False, 1 for True')
parser.add_argument('--tr', default=1, type=int, help='Number of trials to be run starting with seed value entered for seed.')
parser.add_argument('--name', default='.', type=str, help='Special string name.')
parser.add_argument('--noise_scale', default=0, type=float, help='sigma of normally random noise added to test set')
parser.add_argument('--robust', default=None, type=str, help='method of robust training')
parser.add_argument('--adversarial', default=0, type=int, help='0 if using random perturb, 1 if adversarial')
parser.add_argument('--method', default='TS', type=str, help='TS (two-stage learning) or DF (decision-focused learning).')
parser.add_argument('--env', default='general', type=str, help='general (MDP) or POMDP.')
parser.add_argument('--data', default='synthetic', type=str, help='synthetic or pilot')
parser.add_argument('--sv', default='.', type=str, help='save string name')
parser.add_argument('--ope', default='sim', type=str, help='importance sampling (IS) or simulation-based (sim).')
parser.add_argument('--eps', default=0.1, type=float, help='epsilon used for calculating soft top k')


args=parser.parse_args()

args.save=bool(args.save)
args.plot=bool(args.plot)
args.compute=bool(args.compute)

attempts = defaultdict(int)
successes = defaultdict(int)

### Launch new computational experiments for the specified settings if True
for eps in EPSILONS:
    print('EPSILON:', eps)
    args_copy = copy(args)
    args_copy.eps = eps
    for sd in range(args.seed, args.seed+args.tr):
        args_copy.sd = sd
        save_name = f'exp_eps_{eps}'
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
        attempts[eps] += 1
        try:
            train_main(args)
            successes[eps] += 1
            # os.system(f'python3 {curr_dir}/train.py --method DF --sv {df_sim_filename} --epochs {args.epochs} --instances {args.instances} --seed {sd} --ope {"sim"} --noise_scale {args.noise_scale} {robust_clause} --adversarial {args.adversarial} --eps {eps}', shell=True)
        except Exception as e:
            if isinstance(e, ValueError):
                # it failed...
                pass
            elif isinstance(e, IndexError):
                # different error, so skip
                attempts[eps] -= 1

    print('successes:', successes)
    print('attempts:', attempts)

print('finished!')
with open('eps_summary.json', 'w') as f:
    json.dump([successes, attempts], f)