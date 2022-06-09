import numpy as np
import pickle


for noise in [0.0, 0.1, 0.2, 0.3]:
    ts = []
    df = []
    random = []
    optimal = []
    for seed in range(2, 7):
        for loss in ['mse', 'dfl']:
            with open(f'exps/budgetalloc_{loss}_noise_{noise}_seed_{seed}', 'rb') as f:
                data = pickle.load(f)
            if loss == 'mse':
                ts.append(data['test'])
            else:
                df.append(data['test'])
            random.append(data['random'])
            optimal.append(data['optimal'])
        
    ts = np.array(ts)
    df = np.array(df)
    random = np.array(random)
    optimal = np.array(optimal)
    print(ts.mean(), df.mean(), random.mean(), optimal.mean(), sep='\t')