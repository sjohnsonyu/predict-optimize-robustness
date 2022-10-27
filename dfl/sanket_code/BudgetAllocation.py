from PThenO import PThenO
import pickle
import random
import numpy as np
from SubmodularOptimizer import SubmodularOptimizer
import torch

# W = torch.Tensor([[20, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
# W = torch.Tensor([[100, 1, 1, 1, 1],
#                   [10, 10, 10, 10, 10]])
W = torch.Tensor([100, 1, 1, 1, 1])

class BudgetAllocation(PThenO):
    """The budget allocation predict-then-optimise problem from Wilder et. al. (2019)"""

    def __init__(
        self,
        num_train_instances=100,  # number of instances to use from the dataset to train
        num_test_instances=500,  # number of instances to use from the dataset to test
        num_targets=10,  # number of items to choose from
        num_items=5,  # number of targets to consider
        budget=2,  # number of items that can be picked
        num_fake_targets=0,  # number of random features added to make the task harder
        val_frac=0.2,  # fraction of training data reserved for validation
        rand_seed=0,  # for reproducibility
    ):
        super(BudgetAllocation, self).__init__()
        # Do some random seed fu
        self.rand_seed = rand_seed
        self._set_seed(self.rand_seed)
        train_seed, test_seed = random.randrange(2**32), random.randrange(2**32)

        # Load train and test labels
        self.num_train_instances = num_train_instances
        self.num_test_instances = num_test_instances
        Ys_train_test = []
        for seed, num_instances in zip([train_seed, test_seed], [num_train_instances, num_test_instances]):
            # Set seed for reproducibility
            self._set_seed(seed)

            # Load the relevant data (Ys)
            Ys = self._load_instances(num_instances, num_items, num_targets)  # labels
            assert not torch.isnan(Ys).any()

            # Save Xs and Ys
            Ys_train_test.append(Ys)
        self.Ys_train, self.Ys_test = (*Ys_train_test,)

        # Generate features based on the labels
        self.num_targets = num_targets
        self.num_fake_targets = num_fake_targets
        self.num_features = self.num_targets + self.num_fake_targets
        self.Xs_train, self.Xs_test = self._generate_features([self.Ys_train, self.Ys_test], self.num_fake_targets)  # features
        assert not (torch.isnan(self.Xs_train).any() or torch.isnan(self.Xs_test).any())

        # Split training data into train/val
        assert 0 < val_frac < 1
        self.val_frac = val_frac
        self.val_idxs = range(0, int(self.val_frac * num_train_instances))
        self.train_idxs = range(int(self.val_frac * num_train_instances), num_train_instances)
        assert all(x is not None for x in [self.train_idxs, self.val_idxs])

        # Create functions for optimisation
        assert budget < num_items
        self.budget = budget
        self.opt = SubmodularOptimizer(self.get_objective, self.budget)

        # Undo random seed setting
        self._set_seed()


    def _load_instances(self, num_instances, num_items, num_targets):
        """
        Loads the labels (Ys) of the prediction from a file, and returns a subset of it parameterised by instances.
        """
        # Load the dataset
        # curr_dir = '/n/home05/sjohnsonyu/predict-optimize-robustness/dfl/sanket_code'
        curr_dir = './'
        # with open(f'{curr_dir}/data/dummy_0.3.pickle', 'rb') as f:
        with open(f'{curr_dir}/data/budget_allocation_data.pickle', 'rb') as f:
            Yfull, _ = pickle.load(f, encoding='bytes')

        Yfull = np.array(Yfull)
        # print('Setting dummy data')
        # Yfull = np.full(Yfull.shape, 0.3)  # TODO remove me
        # Whittle the dataset down to the right size
        def whittle(matrix, size, dim):
            assert size <= matrix.shape[dim]
            elements = np.random.choice(matrix.shape[dim], size)
            return np.take(matrix, elements, axis=dim)
        Ys = whittle(Yfull, num_instances, 0)
        Ys = whittle(Ys, num_items, 1)
        Ys = whittle(Ys, num_targets, 2)

        return torch.from_numpy(Ys).float().detach()

    def _generate_features(self, Ysets, num_fake_targets):
        """
        Converts labels (Ys) + random noise, to features (Xs)
        """
        # Generate random matrix common to all Ysets (train + test)
        transform_nn = torch.nn.Sequential(torch.nn.Linear(self.num_features, self.num_targets))

        # Generate training data by scrambling the Ys based on this matrix
        Xsets = []
        for Ys in Ysets:
            # Normalise data across the last dimension
            Ys_mean = Ys.reshape((-1, Ys.shape[2])).mean(dim=0)
            Ys_std = Ys.reshape((-1, Ys.shape[2])).std(dim=0)
            Ys_standardised = (Ys - Ys_mean) / (Ys_std + 1e-10)
            assert not torch.isnan(Ys_standardised).any()

            # Add noise to the data to complicate prediction
            fake_features = torch.normal(mean=torch.zeros(Ys.shape[0], Ys.shape[1], num_fake_targets))
            Ys_augmented = torch.cat((Ys_standardised, fake_features), dim=2)

            # Encode Ys as features by multiplying them with a random matrix
            Xs = transform_nn(Ys_augmented).detach().clone()
            Xsets.append(Xs)

        return (*Xsets,)

    def get_train_data(self):
        return self.Xs_train[self.train_idxs], self.Ys_train[self.train_idxs],  [None for _ in range(len(self.train_idxs))]

    def get_val_data(self):
        return self.Xs_train[self.val_idxs], self.Ys_train[self.val_idxs],  [None for _ in range(len(self.val_idxs))]

    def get_test_data(self):
        return self.Xs_test, self.Ys_test,  [None for _ in range(len(self.Ys_test))]

    def get_modelio_shape(self):
        return self.num_features, self.num_targets
    
    def get_output_activation(self):
        return 'relu'

    def get_twostageloss(self):
        return 'mse'

    def get_objective(self, Y, Z, w=W, **kwargs):
        """
        For a given set of predictions/labels (Y), returns the decision quality.
        The objective needs to be _maximised_.
        """
        # Sanity check inputs
        assert Y.shape[-2] == Z.shape[-1]
        assert len(Z.shape) + 1 == len(Y.shape)

        # Initialise weights to default value
        if w is None:
            w = torch.ones(Y.shape[-1]).requires_grad_(False)
        else:
            # assert Y.shape[-1] == w.shape[0]
            # assert len(w.shape) == 1
            pass
        # Calculate objective
        # breakpoint()
        p_fail = 1 - Z.unsqueeze(-1) * Y  # (b, c, u)
        p_all_fail = p_fail.prod(dim=-2)  # (b, u)

        # if len(Y.shape) > 2:
        #     w = w.repeat(Y.shape[0], 1, 1)

        # breakpoint()
        # obj = (w*Z.unsqueeze(-1) * (1 - p_all_fail)).sum(dim=-1)
        obj = (w * (1 - p_all_fail)).sum(dim=-1)

        # obj = (Z.unsqueeze(-1) * Y * w).sum(dim=(-2, -1)) # added term to capture user/channel value
        # obj = ((1 - p_all_fail)).sum(dim=-1)
        # obj = (1 - p_all_fail).sum(dim=-1) + (Z.unsqueeze(-1) * Y * w).sum(dim=(-2, -1)) # added term to capture user/channel value
        return obj

    def get_decision(self, Y, Z_init=None, **kwargs):
        # If this is a single instance of a decision problem
        if len(Y.shape) == 2:
            return self.opt(Y, Z_init=Z_init)

        # If it's not...
        #   Remember the shape
        Y_shape = Y.shape
        #   Break it down into individual instances and solve
        Y_new = Y.view((-1, Y_shape[-2], Y_shape[-1]))
        Z = torch.cat([self.opt(y, Z_init=Z_init) for y in Y_new], dim=0)
        #   Convert it back to the right shape
        Z = Z.view((*Y_shape[:-2], -1))
        return Z

