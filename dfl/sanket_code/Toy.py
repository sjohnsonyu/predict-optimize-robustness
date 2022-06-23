from unittest import result
from PThenO import PThenO
import pickle
import random
import numpy as np
from SubmodularOptimizer import SubmodularOptimizer
import torch
from scipy import optimize

class Toy(PThenO):
    """toy problem"""

    def __init__(
        self,
        num_train_instances=200,  # number of instances to use from the dataset to train
        num_test_instances=100,  # number of instances to use from the dataset to test
        val_frac=0.2,  # fraction of training data reserved for validation
        rand_seed=0,  # for reproducibility
    ):
        super(Toy, self).__init__()
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
            Ys = self._load_instances(num_instances)  # labels
            assert not torch.isnan(Ys).any()

            # Save Xs and Ys
            Ys_train_test.append(Ys)
        self.Ys_train, self.Ys_test = (*Ys_train_test,)

        # dummy data: X = Y
        self.Xs_train, self.Xs_test = self.Ys_train, self.Ys_test
        assert not (torch.isnan(self.Xs_train).any() or torch.isnan(self.Xs_test).any())

        # Split training data into train/val
        assert 0 < val_frac < 1
        self.val_frac = val_frac
        self.val_idxs = range(0, int(self.val_frac * num_train_instances))
        self.train_idxs = range(int(self.val_frac * num_train_instances), num_train_instances)
        assert all(x is not None for x in [self.train_idxs, self.val_idxs])

        # Create functions for optimisation
        # self.opt = SubmodularOptimizer(self.get_objective, self.budget)
        self.offset = self._get_opt_offset()
        # Undo random seed setting
        self._set_seed()

    def _get_opt_offset(self):
        opt_result = optimize.minimize(lambda x: -self._reward_function(x), 0)  # maximize
        return opt_result.x[0]

    def _reward_function(self, x):
        if torch.is_tensor(x):
            first = 1 / (1 + torch.exp(-4*x))
            second = -1 / (1 + 25*torch.exp(-x)) / 10
            return first + second
        first = 1 / (1 + np.exp(-4*x))
        second = -1 / (1 + 25*np.exp(-x)) / 10
        return first + second

    def _load_instances(self, num_instances):
        """
        Creates a random dataset and returns a subset of it parameterised by instances.
        """
         # TODO: what ranges do we want to allow?
        Yfull = np.random.randint(-10, 11, size=(10000,1)) 
        # Whittle the dataset down to the right size
        def whittle(matrix, size, dim):
            assert size <= matrix.shape[dim]
            elements = np.random.choice(matrix.shape[dim], size)
            return np.take(matrix, elements, axis=dim)
        Ys = whittle(Yfull, num_instances, 0)
        return torch.from_numpy(Ys).float().detach()

    def get_train_data(self):
        return self.Xs_train[self.train_idxs], self.Ys_train[self.train_idxs],  [None for _ in range(len(self.train_idxs))]

    def get_val_data(self):
        return self.Xs_train[self.val_idxs], self.Ys_train[self.val_idxs],  [None for _ in range(len(self.val_idxs))]

    def get_test_data(self):
        return self.Xs_test, self.Ys_test,  [None for _ in range(len(self.Ys_test))]

    def get_modelio_shape(self):
        return 1, 1  # x, y dims
    
    def get_output_activation(self):
        return 'relu'

    def get_twostageloss(self):
        return 'mse'

    def get_optimal_z(self, y):
        return y + self.offset

    def get_objective(self, Y, Z, **kwargs):
        """
        For a given set of predictions/labels (Y), returns the decision quality.
        The objective needs to be _maximised_.
        """
        # Sanity check inputs
        # assert Y.shape[-2] == Z.shape[-1]
        # assert len(Z.shape) + 1 == len(Y.shape)
        # TODO: may need to be vectorized

        return self._reward_function(Z - Y).squeeze()

    def get_decision(self, Y, **kwargs):
        # If this is a single instance of a decision problem
        if len(Y.shape) == 2:
            return self.get_optimal_z(Y)
        # If it's not...
        # Y_shape = Y.shape
        #   Break it down into individual instances and solve
        # Y_new = Y#.view((-1, Y_shape[-2], Y_shape[-1]))
        Z = torch.Tensor([self.get_optimal_z(y) for y in Y]).unsqueeze(1)
        #   Convert it back to the right shape
        # Z = Z.view((*Y_shape[:-2], -1))
        return Z

