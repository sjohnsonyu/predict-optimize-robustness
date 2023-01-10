from PThenO import PThenO
import torch
import cvxpy as cp
import numpy as np
from cvxpylayers.torch import CvxpyLayer


class BabyPortfolioOpt(PThenO):

    def __init__(
        self,
        num_train_instances=200,
        num_test_instances=200,
        x_dim=10,
        num_stocks=5,  # number of stocks per instance to choose from
        val_frac=0.2,  # fraction of training data reserved for validating
        rand_seed=0,  # for reproducibility
        alpha=1,  # risk aversion constant
        num_fake_targets=0,
        synthetic_hidden_dim=16,
        num_synthetic_layers=1
    ):
        super(BabyPortfolioOpt, self).__init__()
        # Do some random seed fu
        self.rand_seed = rand_seed
        self._set_seed(self.rand_seed)

        self.num_train_instances = num_train_instances
        self.num_test_instances = num_test_instances

        # Load train and test labels
        self.num_stocks = num_stocks
        self.Ys = None
        self.Xs = None
        self.x_dim = x_dim
        self.hidden_dim = synthetic_hidden_dim
        self.num_fake_targets = num_fake_targets
        self.num_synthetic_layers = num_synthetic_layers
        self._set_dummy_data()

        # Split data into train/val/test
        # Sanity check and initialisations
        assert 0 < val_frac < 1
        self.val_frac = val_frac

        # Creating splits for train/valid/test
        num_val = int(self.val_frac * self.num_train_instances)
        idxs = np.array(list(range(self.num_train_instances + self.num_test_instances)))
        self.train_idxs = idxs[:self.num_train_instances - num_val]
        self.val_idxs = idxs[self.num_train_instances - num_val:self.num_train_instances]
        self.test_idxs = idxs[self.num_train_instances:]
        assert all(x is not None for x in [self.train_idxs, self.val_idxs, self.test_idxs])

        # Create functions for optimisation
        # TODO: Try larger constant
        self.alpha = alpha
        self.opt = self._create_cvxpy_problem(alpha=self.alpha)

        # Undo random seed setting
        self._set_seed()

    def _set_dummy_data(self):
        num_examples = self.num_test_instances + self.num_test_instances
        self.Ys = torch.Tensor(np.random.random((num_examples, self.num_stocks)))
        self.Xs = self._generate_features(self.Ys)
        # print('copying Ys to Xs')
        # self.Xs = self.Ys.clone().unsqueeze(2) * -0.33
        identity = torch.eye(self.num_stocks).unsqueeze(dim=0)
        self.covar_mat = identity.repeat(num_examples, 1, 1)

    def _create_cvxpy_problem(self, alpha):
        x_var = cp.Variable(self.num_stocks)
        L_sqrt_para = cp.Parameter((self.num_stocks, self.num_stocks))
        p_para = cp.Parameter(self.num_stocks)
        constraints = [x_var >= 0, x_var <= 1, cp.sum(x_var) == 1]
        objective = cp.Maximize(p_para.T @ x_var - alpha * cp.sum_squares(L_sqrt_para @ x_var))
        problem = cp.Problem(objective, constraints)
        return CvxpyLayer(problem, parameters=[p_para, L_sqrt_para], variables=[x_var])

    def get_train_data(self, **kwargs):
        return self.Xs[self.train_idxs], self.Ys[self.train_idxs], self.covar_mat[self.train_idxs]

    def get_val_data(self, **kwargs):
        return self.Xs[self.val_idxs], self.Ys[self.val_idxs], self.covar_mat[self.val_idxs]

    def get_test_data(self, **kwargs):
        return self.Xs[self.test_idxs], self.Ys[self.test_idxs], self.covar_mat[self.test_idxs]

    def get_modelio_shape(self):
        return self.Xs.shape[-1], 1

    def get_twostageloss(self):
        return 'mse'

    def _create_feature_generator(self):
        layers = []
        input_size = 1 + self.num_fake_targets
        hidden_size = self.hidden_dim
        output_size = self.x_dim

        if self.num_synthetic_layers == 1:  # no hidden layer
            layers.append(torch.nn.Linear(input_size, output_size))
        else:
            layers.append(torch.nn.Linear(input_size, hidden_size))
            layers.append(torch.nn.ReLU())

            for _ in range(self.num_synthetic_layers - 2):
                layers.append(torch.nn.Linear(hidden_size, hidden_size))
                layers.append(torch.nn.ReLU())

            layers.append(torch.nn.Linear(hidden_size, output_size))

        return torch.nn.Sequential(*layers)


    def _generate_features(self, Ys):
        """
        Converts labels (Ys) + random noise, to features (Xs)
        """
        # Generate random matrix common to all Ysets (train + test)
        # self.feature_generator = torch.nn.Sequential(torch.nn.Linear(1, self.x_dim))  # TODO double check this!
        self.feature_generator = self._create_feature_generator()
        # Generate training data by scrambling the Ys based on this matrix
        # Normalise data across the last dimension
        Ys_mean = Ys.mean(dim=0)
        Ys_std = Ys.std(dim=0)
        Ys_standardised = (Ys - Ys_mean) / (Ys_std + 1e-10)
        # assert not torch.isnan(Ys_standardised).any()
        # Ys_standardised = Ys.clone()

        # Add noise to the data to complicate prediction
        Ys_standardised_flattened = Ys_standardised.flatten().unsqueeze(1)

        fake_features = torch.normal(mean=torch.zeros(Ys_standardised_flattened.shape[0], self.num_fake_targets))
        Ys_augmented = torch.cat((Ys_standardised_flattened, fake_features), dim=1)


        # Encode Ys as features by multiplying them with a random matrix
        Xs = self.feature_generator(Ys_augmented).detach()
        Xs = Xs.reshape(len(Ys), self.num_stocks, self.x_dim)

        return Xs

    def get_decision(self, Y, aux_data, max_instances_per_batch=1500, **kwargs):
        # Get the sqrt of the covariance matrix
        covar_mat = aux_data
        sqrt_covar = torch.linalg.cholesky(covar_mat)

        # Split Y into reasonably sized chunks so that we don't run into memory issues
        # Assumption Y is only 2D at max
        assert Y.ndim <= 2
        # if Y.ndim == 2:
        #     results = []
        #     for start in range(0, Y.shape[0], max_instances_per_batch):
        #         end = min(Y.shape[0], start + max_instances_per_batch)
        #         result = self.opt(Y[start:end], sqrt_covar)[0]
        #         results.append(result)
        #     return torch.cat(results, dim=0)
        # else:
        return self.opt(Y, sqrt_covar)[0]

    def get_objective(self, Y, Z, aux_data, **kwargs):
        # TODO: look at either torch.bmm or torch.matmul
        covar_mat = aux_data
        covar_mat_Z_t = (torch.linalg.cholesky(covar_mat) * Z.unsqueeze(dim=-2)).sum(dim=-1)
        quad_term = covar_mat_Z_t.square().sum(dim=-1)
        obj = (Y * Z).sum(dim=-1) - self.alpha * quad_term
        return obj
    
    def get_output_activation(self):
        return 'tanh'


if __name__ == "__main__":
    problem = BabyPortfolioOpt()
    X_train, Y_train, Y_train_aux = problem.get_train_data()
    Z_train = problem.get_decision(Y_train, aux_data=Y_train_aux)
    obj = problem.get_objective(Y_train, Z_train, aux_data=Y_train_aux)
