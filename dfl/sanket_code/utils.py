import os
import pdb
from pickletools import optimize
from typing import Dict
import pandas as pd
import pickle
import torch
import inspect
from itertools import repeat
from scipy import optimize, rand
import numpy as np
from Toy import Toy
from ToyMod import ToyMod
from BabyPortfolioOpt import BabyPortfolioOpt
from BudgetAllocation import BudgetAllocation
import qpth


def init_if_not_saved(
    problem_cls,
    kwargs,
    folder='./',
    # folder='/n/home05/sjohnsonyu/predict-optimize-robustness/dfl/sanket_code/models',
    load_new=True,
):
    # Find the filename if a saved version of the problem with the same kwargs exists
    master_filename = os.path.join(folder, f"{problem_cls.__name__}.csv")
    filename, saved_probs = find_saved_problem(master_filename, kwargs)
 
    if not load_new and filename is not None:
        # Load the model
        with open(filename, 'rb') as file:
            problem = pickle.load(file)
    else:
        # Initialise model from scratch
        problem = problem_cls(**kwargs)

        # Save model for the future
        print("Saving the problem")
        filename = os.path.join(folder, f"{problem_cls.__name__}_{len(saved_probs)}.pkl")
        with open(filename, 'wb') as file:
            pickle.dump(problem, file)

        # Add its details to the master file
        kwargs['filename'] = filename
        saved_probs = saved_probs.append([kwargs])
        with open(master_filename, 'w') as file:
            saved_probs.to_csv(file, index=False)

    return problem

def find_saved_problem(
    master_filename: str,
    kwargs: Dict,
):
    # Open the master file with details about saved models
    if os.path.exists(master_filename):
        with open(master_filename, 'r') as file:
            saved_probs = pd.read_csv(file)
    else:
        saved_probs = pd.DataFrame(columns=('filename', *kwargs.keys(),))
    
    # Check if the problem has been saved before
    relevant_models = saved_probs
    for col, val in kwargs.items():
        if col in relevant_models.columns:
            relevant_models = relevant_models.loc[relevant_models[col] == val]  # filtering models by parameters

    # If it has, find the relevant filename
    filename = None
    if not relevant_models.empty:
        filename = relevant_models['filename'].values[0]
    
    return filename, saved_probs


def add_noise(y, problem, Zs, aux_data=None, scale=0, noise_type='random', low=0, high=1, adv_backprop=0):
    if isinstance(problem, Toy) or isinstance(problem, ToyMod):
        low = problem.get_low()
        high = problem.get_high()
    elif isinstance(problem, BabyPortfolioOpt):
        low = None
        high = None
    else:
        low = 0
        high = 1

    if noise_type == 'random':
        return add_random_noise(torch.clone(y), scale, low=low, high=high)
    elif noise_type == 'adversarial':
        # NOTE: I removed the y.clone() here, want to sanity check
        return add_adversarial_noise(y, problem, Zs, aux_data=aux_data, budget=scale, low=low, high=high, adv_backprop=adv_backprop)
    else:
        raise Exception("noise type is not valid. please select either random or adversarial")


def add_random_noise(y, scale, low=0, high=1):
    skip_clipping = low is None or high is None
    if isinstance(y, torch.Tensor) and len(y.shape) == 3:
        batch_sz, num_channels, num_users = y.shape
        noise = np.random.randn(batch_sz, num_channels, num_users) * scale
        noisy_y = y + noise
        noisy_y = clip(noisy_y, low, high) if not skip_clipping else noisy_y
        return noisy_y
    elif isinstance(y, torch.Tensor) and len(y.shape) == 2:
        batch_sz, num_dims = y.shape
        noise = np.random.randn(batch_sz, num_dims) * scale
        noisy_y = y + noise
        noisy_y = clip(noisy_y, low, high) if not skip_clipping else noisy_y
        return noisy_y
    else:
        return y


def add_adversarial_noise(y,
                          problem,
                          Zs,
                          aux_data=None,
                          budget=100,
                          lr=1e-2,
                          norm=1,
                        #   norm=2,
                          num_iters=30,
                          low=0,
                          high=1,
                          num_random_inits=10,
                          random_init_scale=3,
                          random_init_bias=2,
                          adv_backprop=0,
                         ):
    if not isinstance(y, torch.Tensor): return y
    if isinstance(problem, BudgetAllocation) and len(y.shape) != 3: return y

    no_random_inits = num_random_inits is None
    num_random_inits = 1 if num_random_inits is None else num_random_inits

    if isinstance(problem, Toy) or isinstance(problem, ToyMod):
        all_perturbed_ys = torch.zeros(y.shape[0], num_random_inits)
        all_perturbed_rewards = torch.zeros(y.shape[0], num_random_inits)
    elif isinstance(problem, BabyPortfolioOpt):
        all_perturbed_ys = torch.zeros(y.shape[0], y.shape[1], num_random_inits)
        all_perturbed_rewards = torch.zeros(y.shape[0], num_random_inits)
    else:  # budget allocation
        all_perturbed_ys = torch.zeros(y.shape[0], y.shape[1], y.shape[2], num_random_inits)
        all_perturbed_rewards = torch.zeros(y.shape[0], num_random_inits)

    # TODO break into separate function
    for i in range(num_random_inits):
        if no_random_inits:
            perturbed_y = y.clone()
        else:
            if isinstance(problem, Toy) or isinstance(problem, ToyMod):
                perturbed_y = y.clone() + (random_init_scale * torch.randn(y.shape[0])).unsqueeze(1) + random_init_bias # (want Z - Y to be smaller than offset)
            else:
                perturbed_y = y.clone() + (random_init_scale * torch.randn(y.shape)) + random_init_bias # (want Z - Y to be smaller than offset)

        perturbed_y = projection(perturbed_y, y, budget=budget, norm=norm, low=low, high=high)
        perturbed_y.requires_grad = True

        optim = torch.optim.SGD([perturbed_y], lr=lr, momentum=0.99)

        for _ in range(num_iters):
            perturbed_rewards = problem.get_objective(perturbed_y, Zs, aux_data=aux_data)  # TODO pass in 0
            perturbed_reward = perturbed_rewards.sum()
            optim.zero_grad()
            perturbed_reward.backward()
            optim.step()
            with torch.no_grad():
                perturbed_y = projection(perturbed_y, y, budget=budget, norm=norm, low=low, high=high)

        if isinstance(problem, Toy) or isinstance(problem, ToyMod):
            all_perturbed_ys[:, i] = perturbed_y.squeeze()
            all_perturbed_rewards[:, i] = perturbed_rewards
        elif isinstance(problem, BabyPortfolioOpt):
            all_perturbed_ys[:, :, i] = perturbed_y  # TODO not sure if this is the right thing to do
            all_perturbed_rewards[:, i] = perturbed_rewards
        else:
            all_perturbed_ys[:, :, :, i] = perturbed_y
            all_perturbed_rewards[:, i] = perturbed_rewards

    # all_perturbed_ys = all_perturbed_ys
    idxs = torch.argmin(all_perturbed_rewards, dim=1)
    if isinstance(problem, Toy) or isinstance(problem, ToyMod):
        perturbed_y = all_perturbed_ys[range(len(idxs)), idxs].unsqueeze(1).detach()
        perturbed_rewards = problem.get_objective(perturbed_y, Zs)
    elif isinstance(problem, BabyPortfolioOpt):
        perturbed_y = all_perturbed_ys[range(len(idxs)), :, idxs].detach()
        perturbed_rewards = problem.get_objective(perturbed_y, Zs, aux_data)  # TODO pass in 0
    else:
        perturbed_y = all_perturbed_ys[range(len(idxs)), :, :, idxs].detach()
        perturbed_rewards = problem.get_objective(perturbed_y, Zs)


    perturbed_reward = perturbed_rewards.sum()
    if adv_backprop == 0:
        return perturbed_y

    # df_dzs = torch.zeros(Zs.shape[0], 1)
    # z_var = Zs[i].detach().requires_grad_(True)
    # df_dzs[i] = torch.autograd.grad(perturbed_reward_var, z_var, retain_graph=True, create_graph=True)[0] 
    # how would we incorporate df_dzs in the backward pass?

    final_perturbed_ys = torch.zeros(y.shape[0], 1)
    for i in range(len(perturbed_y)):
        # Differentiable part!
        Q = torch.eye(len(perturbed_y[i])) # typically Hessian, but sub for arbitrary SPD matrix
        lower_bound = abs(max(y[i] - budget, low))
        upper_bound = abs(min(y[i] + budget, high))
        # Gx <= b
        A, b, G, h = torch.Tensor([]), torch.Tensor([]), torch.Tensor([[-1], [1]]), torch.Tensor([lower_bound, upper_bound]) # constraint matrix
        y_var = y[i].requires_grad_(True)
        perturbation_var = (perturbed_y[i].detach() - y[i]).requires_grad_(True)
        reward_var = problem.get_objective(y_var + perturbation_var, Zs[i], aux_data=aux_data[i])
        jac = torch.autograd.grad(reward_var, perturbation_var, retain_graph=True, create_graph=True)[0] # TODO double check
        p = jac - Q * (y_var + perturbation_var)
        qp_solver = qpth.qp.QPFunction(verbose=-1)
        approx_y = qp_solver(Q, p, G, h, A, b)[0]

        final_perturbed_ys[i] = approx_y

    return final_perturbed_ys


def clip(tensor, low, high):
    too_low_mask = tensor < low
    too_high_mask = tensor > high
    tensor[too_low_mask] = low
    tensor[too_high_mask] = high
    return tensor


def projection(perturbed_y, y, budget=1, norm=1, low=0, high=1):
    skip_clipping = low is None or high is None
    perturbed_y = clip(perturbed_y, low, high) if not skip_clipping else perturbed_y
    if len(perturbed_y.shape) == 3:
        batch_sz, num_channels, num_users = y.shape
        for i in range(batch_sz):
            perturbation = (perturbed_y[i] - y[i])

            perturbation_norm = torch.norm(perturbation.flatten(), p=norm)
            if perturbation_norm > budget:
                # FIXME: I'm confused... should the 0.1 be spread across? I think that's how
                # we did it previously, but that's not in line with the proj inf.
                if norm == float('inf'):
                    perturbation = clip(perturbation, -budget, budget)
                    perturbed_y[i] = y[i] + perturbation
                elif norm == 1 or norm == 2:
                    perturbed_y[i] = perturbed_y[i] - perturbation + perturbation / perturbation_norm * budget

        # buggy: reshape kills the gradient!
        # perturbed_y = perturbed_y.reshape(batch_sz, num_channels * num_users)
        # y = y.reshape(batch_sz, num_channels * num_users)
        # perturbation = (perturbed_y - y).reshape(batch_sz, num_channels * num_users)
        # perturbation_norm = torch.norm(perturbation, p=norm, dim=1)
        # mask = perturbation_norm > budget
        # if sum(mask) > 0:
        #     perturbed_y[mask] = perturbed_y[mask] - perturbation[mask] + perturbation[mask] / perturbation_norm[mask].unsqueeze(1) * budget
        # return perturbed_y.reshape(batch_sz, num_channels, num_users)

        return perturbed_y
    elif len(perturbed_y.shape) == 2:
        perturbation = perturbed_y - y
        perturbation_norm = torch.norm(perturbation, p=norm, dim=1)
        if norm == 1:
            mask = perturbation_norm > budget
            if sum(mask) > 0:
                perturbed_y[mask] = perturbed_y[mask] - perturbation[mask] + perturbation[mask] / perturbation_norm[mask].unsqueeze(1) * budget
        elif norm == 2:
            mask = perturbation_norm > budget
            if sum(mask) > 0:
                perturbed_y[mask] = perturbed_y[mask] - perturbation[mask] + perturbation[mask] / perturbation_norm[mask].unsqueeze(1) * budget
        elif norm == float('inf'):
            # or y + clip(perturbation)... don't know what we need for gradient
            mask = torch.logical_or(perturbation_norm > budget, perturbation_norm < -budget)
            perturbed_y[mask] = perturbed_y[mask] - perturbation[mask] + clip(perturbation, -budget, budget)[mask]
        return perturbed_y


def print_metrics(
    datasets,
    model,
    problem,
    loss_type,
    loss_fn,
    prefix="",
    noise_type=None,
    add_train_noise=False,
    noise_scale=0,
    adv_backprop=0,
    out_filename=None
):
    metrics = {}
    for Xs, Ys, Ys_aux, partition in datasets:
        # Choose whether we should use train or test 
        isTrain = (partition=='train') and (prefix != "Final")
        # Decision Quality
        pred = model(Xs).squeeze()
        if isinstance(problem, Toy) or isinstance(problem, ToyMod):  # this is a hack; toy is the only domain with preds of dim 1
            pred = pred.unsqueeze(1)
        Zs_pred = problem.get_decision(pred, aux_data=Ys_aux, isTrain=isTrain)
        if partition == 'test' or add_train_noise:
            Ys = add_noise(Ys, problem, Zs_pred.detach(), aux_data=Ys_aux, scale=noise_scale, noise_type=noise_type, adv_backprop=adv_backprop)

        objectives = problem.get_objective(Ys, Zs_pred, aux_data=Ys_aux)  # TODO pass in 0 for Q
        mse_error = (pred - Ys).square().mean()
        # Loss and Error
        # if partition!='test':
        losses = []
        for i in range(len(Xs)):
            # Surrogate Loss
            pred = model(Xs[i])
            if not isinstance(problem, Toy) and not isinstance(problem, ToyMod):
                pred = pred.squeeze()
            losses.append(loss_fn(pred, Ys[i], aux_data=Ys_aux[i], partition=partition, index=i))
        losses = torch.stack(losses).flatten()
        # else:
        #     losses = torch.zeros_like(objectives)

        # Print
        objective = objectives.mean().item()
        loss = losses.mean().item()
        loss_live = losses.mean()
        mae = torch.nn.L1Loss()(losses, -objectives).item()
        print(f"{prefix} {partition} DQ: {objective}, Loss: {loss}, MSE: {mse_error}")
        metrics[partition] = {'objective': objective, 'loss': loss, 'mae': mae, 'loss_live': loss_live, 'mse': mse_error}

    return metrics, Ys

def starmap_with_kwargs(pool, fn, args_iter, kwargs):
    args_for_starmap = zip(repeat(fn), args_iter, repeat(kwargs))
    return pool.starmap(apply_args_and_kwargs, args_for_starmap)

def apply_args_and_kwargs(fn, args, kwargs):
    return fn(*args, **kwargs)

def gather_incomplete_left(tensor, I):
    return tensor.gather(I.ndim, I[(...,) + (None,) * (tensor.ndim - I.ndim)].expand((-1,) * (I.ndim + 1) + tensor.shape[I.ndim + 1:])).squeeze(I.ndim)

def trim_left(tensor):
    while tensor.shape[0] == 1:
        tensor = tensor[0]
    return tensor

class View(torch.nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def __repr__(self):
        return f'View{self.shape}'

    def forward(self, input):
        '''
        Reshapes the input according to the shape saved in the view data structure.
        '''
        batch_size = input.shape[:-1]
        shape = (*batch_size, *self.shape)
        out = input.view(shape)
        return out

def solve_lineqn(A, b, eps=1e-5):
    try:
        result = torch.linalg.solve(A, b)
    except RuntimeError:
        print(f"WARNING: The matrix was singular")
        result = torch.linalg.solve(A + eps * torch.eye(A.shape[-1]), b)
    return result

def move_to_gpu(problem):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for key, value in inspect.getmembers(problem, lambda a:not(inspect.isroutine(a))):
        if isinstance(value, torch.Tensor):
            problem.__dict__[key] = value.to(device)
