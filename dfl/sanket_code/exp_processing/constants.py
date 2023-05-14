
BA_WEIGHTS = [0.0, 0.025, 0.05, 0.075, 0.1, 0.125]
PO_WEIGHTS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
DP_WEIGHTS = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
NUM_DECIMAL_PLACES = 5
DOMAIN_TO_WEIGHTS = {'toy': DP_WEIGHTS,
                     'babyportfolio': PO_WEIGHTS,
                     'budgetalloc': BA_WEIGHTS,
                     'toymod': DP_WEIGHTS,
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
                  'toymod': "Demand Prediction",
                 }

DOMAIN_TO_DEFAULT_X_DIM = {'toy': 10,
                            'babyportfolio': 10,
                            'budgetalloc': 5,
                            'toymod': 10
                          }
DOMAIN_TO_X_DIM_OPTIONS = {'toy': [1, 2, 5, 10],
                           'babyportfolio': [2, 5, 10],
                           'budgetalloc': [2, 5, 10],
                           'toymod': [1, 2, 5, 10]
                 }
FAKE_TARGET_OPTIONS = [0, 5, 10, 20]

DOMAIN_TO_MILD_NOISE = {'toy': 1.0,
                        'babyportfolio': 0.1,
                        'budgetalloc': 0.025,
                        'toy': 1.0,
                       }