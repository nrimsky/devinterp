import torch as t
from tqdm import tqdm
import random
from helpers import eval_model, get_submodule_param_mask
from math import log, sqrt
from train import ExperimentParams
from model import MLP
from dataset import make_dataset, train_test_split, make_random_dataset
from matplotlib import pyplot as plt
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable
import json


@dataclass
class SGLDParams:
    gamma: float = 1
    epsilon: float = 0.001
    n_steps: int = 10000
    m: int = 512
    restrict_to_orth_grad: bool = False
    get_updated_model_parameters: Callable = lambda model: model.parameters()
    log_loss_multiplier: float = 1


def cross_entropy_loss(logits, y_s):
    """
    logits: outputs of model
    y: target labels

    returns: mean cross entropy loss
    """
    preds = t.nn.functional.softmax(logits, dim=1)
    return -1 * t.mean(t.log(preds[t.arange(len(preds)), y_s] + 1e-6))


def sgld(model, sgld_params, dataset, beta, device):
    """
    model: MLP model
    sgld_params: SGLDParams object
    dataset: dataset to train on
    beta: temperature parameter

    returns: updated model, lambda_hat
    """
    n = len(dataset)
    model = model.to(device)

    init_loss = eval_model(model, dataset, device)
    n_ln_wstar = n * init_loss
    idx = list(range(len(dataset)))
    optimizer = optimizer = t.optim.SGD(
        sgld_params.get_updated_model_parameters(model),
        weight_decay=0,
        lr=1,
    )

    submodule_param_mask = get_submodule_param_mask(model, sgld_params.get_updated_model_parameters).to(device)

    X_1 = t.stack([dataset[b][0][0] for b in range(len(dataset))]).to(device)
    X_2 = t.stack([dataset[b][0][1] for b in range(len(dataset))]).to(device)
    Y = t.stack([dataset[b][1] for b in range(len(dataset))]).to(device)
    w_0 = t.nn.utils.parameters_to_vector(model.parameters()).detach().clone().to(device)

    out = model(X_1, X_2)
    cross_entropy_loss_value = cross_entropy_loss(out, Y)

    # Compute gradients using torch.autograd.grad
    gradients = t.autograd.grad(cross_entropy_loss_value, model.parameters(), create_graph=True)
    ce_loss_grad_w0 = t.nn.utils.parameters_to_vector(gradients).detach().clone().to(device)
    ce_loss_grad_w0 *= submodule_param_mask
    ce_loss_grad_w0 /= ce_loss_grad_w0.norm(p=2)
    optimizer.zero_grad()

    array_loss = []
    array_weight_norm = []
    full_losses = []

    for _ in tqdm(range(sgld_params.n_steps)):
        batch_idx = random.choices(idx, k=sgld_params.m)
        X_1 = t.stack([dataset[b][0][0] for b in batch_idx]).to(device)
        X_2 = t.stack([dataset[b][0][1] for b in batch_idx]).to(device)
        Y = t.stack([dataset[b][1] for b in batch_idx]).to(device)
        optimizer.zero_grad()
        out = model(X_1, X_2)
        cross_entropy_loss_value = cross_entropy_loss(out, Y)
        array_loss.append(cross_entropy_loss_value.item())
        w = t.nn.utils.parameters_to_vector(model.parameters())
        array_weight_norm.append((w * submodule_param_mask).norm(p=2).item())
        elasticity_loss_term = (sgld_params.gamma / 2) * t.sum(((w_0 - w) ** 2))
        log_likelihood_loss_term = cross_entropy_loss_value * n * beta * sgld_params.log_loss_multiplier
        full_loss = (sgld_params.epsilon / 2) * (
            elasticity_loss_term + log_likelihood_loss_term
        )
        full_losses.append((elasticity_loss_term.item(), log_likelihood_loss_term.item()))
        full_loss.backward()
        optimizer.step()
        eta = t.randn_like(w, device=device) * sqrt(sgld_params.epsilon) * submodule_param_mask
        with t.no_grad():
            new_params = t.nn.utils.parameters_to_vector(model.parameters()) + eta
            if sgld_params.restrict_to_orth_grad:
                diff = new_params - w_0
                proj_diff = diff - t.dot(diff, ce_loss_grad_w0) * ce_loss_grad_w0
                new_params = w_0 + proj_diff
            t.nn.utils.vector_to_parameters(new_params, model.parameters())
    wbic = n * sum(array_loss) / len(array_loss)
    lambda_hat = (wbic - n_ln_wstar) / log(n)
    print(f"lambda_hat: {lambda_hat}")
    print(f"wbic: {wbic}")
    print(f"n_ln_wstar: {n_ln_wstar}")
    print(f"init_loss: {init_loss}")
    print(f"sgld_params: {sgld_params}")
    print(f"array_loss: {array_loss[::len(array_loss)//20]}")
    print(f"array_weight_norm: {array_weight_norm[::len(array_weight_norm)//20]}")
    print(f"full_losses: {full_losses[::len(full_losses)//20]}")
    return model, lambda_hat


def hyperparameter_search(
    params_modular_addition_file,
    params_random_file,
    n_steps,
    m,
    epsilon_range,
    gamma_range,
):
    sgld_params = SGLDParams()
    sgld_params.m = m
    params_modular_addition = ExperimentParams.load_from_file(
        params_modular_addition_file
    )
    params_random = ExperimentParams.load_from_file(params_random_file)
    random_dataset = make_random_dataset(params_random.p, params_random.random_seed)
    modular_addition_dataset = make_dataset(params_modular_addition.p)
    random_dataset, _ = train_test_split(
        random_dataset, params_random.train_frac, params_random.random_seed
    )
    modular_addition_dataset, _ = train_test_split(
        modular_addition_dataset,
        params_modular_addition.train_frac,
        params_modular_addition.random_seed,
    )
    beta = 1 / log(len(modular_addition_dataset))
    results_random = defaultdict(list)
    results_modular_addition = defaultdict(list)
    for epsilon in epsilon_range:
        actual_n_steps = int(n_steps / epsilon)
        sgld_params.epsilon = epsilon
        sgld_params.n_steps = actual_n_steps
        for gamma in gamma_range:
            mlp_modular_addition = MLP(params_modular_addition)
            mlp_random = MLP(params_random)
            sgld_params.gamma = gamma
            mlp_modular_addition.load_state_dict(
                t.load(f"models/model_{params_modular_addition.get_suffix()}.pt")
            )
            mlp_random.load_state_dict(
                t.load(f"models/model_{params_random.get_suffix()}.pt")
            )
            _, lambda_hat_modular_addition = sgld(
                mlp_modular_addition,
                sgld_params,
                modular_addition_dataset,
                beta,
                params_modular_addition.device,
            )
            _, lambda_hat_random = sgld(
                mlp_random,
                sgld_params,
                random_dataset,
                beta,
                params_random.device,
            )
            results_modular_addition[epsilon].append(lambda_hat_modular_addition)
            results_random[epsilon].append(lambda_hat_random)
    # plot results
    for epsilon in epsilon_range:
        plt.clf()
        plt.figure()
        plt.plot(
            gamma_range, results_modular_addition[epsilon], label="modular addition"
        )
        plt.plot(gamma_range, results_random[epsilon], label="random")
        plt.title(
            f"$\lambda$ vs $\gamma$ ($\epsilon$={epsilon}, n_steps={n_steps}, m={m})"
        )
        plt.xlabel("$\gamma$")
        plt.ylabel("$\hat{\lambda}$")
        plt.legend()
        plt.savefig(
            f'plots/lambda_vs_gamma_epsilon_{epsilon}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        )
        plt.close()
    for idx, gamma in enumerate(gamma_range):
        plt.clf()
        plt.figure()
        results_per_epsilon_modular_addition = [
            results_modular_addition[epsilon][idx] for epsilon in epsilon_range
        ]
        results_per_epsilon_random = [
            results_random[epsilon][idx] for epsilon in epsilon_range
        ]
        plt.plot(
            epsilon_range,
            results_per_epsilon_modular_addition,
            label="modular addition",
        )
        plt.plot(epsilon_range, results_per_epsilon_random, label="random")
        plt.title(
            f"$\lambda$ vs $\epsilon$ ($\gamma$={gamma}, n_steps={n_steps}, m={m})"
        )
        plt.xlabel("$\epsilon$")
        plt.ylabel("$\hat{\lambda}$")
        plt.legend()
        plt.savefig(
            f'plots/lambda_vs_epsilon_gamma_{gamma}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        )
        plt.close()


def get_lambda(params, sgld_params, checkpoint_no=None):
    model = MLP(params)
    if checkpoint_no is None:
        model.load_state_dict(t.load(f"models/model_{params.get_suffix()}.pt"))
    else:
        model.load_state_dict(
            t.load(
                f"models/checkpoints/{params.get_suffix(checkpoint_no=checkpoint_no)}.pt"
            )
        )
    if params.use_random_dataset:
        dataset = make_random_dataset(params.p, params.random_seed)
    else:
        dataset = make_dataset(params.p)
    train_data, _ = train_test_split(dataset, params.train_frac, params.random_seed)
    beta = 1 / log(len(train_data))
    _, lambda_hat = sgld(model, sgld_params, train_data, beta, params.device)
    return lambda_hat

def get_lambda_per_quantity(param_files, sgld_params, resample=True):
    lambda_values = []
    for param_file in param_files:
        params = ExperimentParams.load_from_file(param_file)
        if not resample and params.lambda_hat is not None:
            lambda_values.append(params.lambda_hat)
            continue
        lambda_hat = get_lambda(params, sgld_params)
        lambda_values.append(lambda_hat)
        param_dict = params.get_dict()
        param_dict["lambda_hat"] = lambda_hat.item()
        with open(param_file, "w") as f:
            json.dump(param_dict, f)
    return lambda_values

def plot_lambda_per_quantity(param_files, quantity_values, quantity_name, sgld_params):
    lambda_values = get_lambda_per_quantity(param_files, sgld_params)
    plt.clf()
    plt.figure()
    plt.plot(quantity_values, lambda_values, marker="o")
    plt.title(f"$\lambda$ vs {quantity_name}")
    plt.xlabel(quantity_name)
    plt.ylabel("$\hat{\lambda}$")
    plt.savefig(
        f'plots/lambda_vs_{quantity_name.replace(" ", "_")}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    )
    plt.close()


def plot_lambda_per_checkpoint(param_file, sgld_params, checkpoints=None):
    lambda_values = []
    params = ExperimentParams.load_from_file(param_file)
    check_list = list(range(params.n_save_model_checkpoints))
    if checkpoints is not None:
        check_list = checkpoints
    for i in check_list:
        lambda_values.append(get_lambda(params, sgld_params, checkpoint_no=i))
    plt.clf()
    plt.figure()
    plt.plot(check_list, lambda_values, marker="o")
    plt.title(f"$\lambda$ vs checkpoint")
    plt.xlabel("checkpoint")
    plt.ylabel("$\hat{\lambda}$")
    plt.savefig(
        f'plots/lambda_vs_checkpoint_{datetime.now().strftime("%Y%m%d_%H%M%S")}_restrictorth{sgld_params.restrict_to_orth_grad}.png'
    )
    plt.close()


if __name__ == "__main__":
    sgld_params = SGLDParams(
        gamma=5,
        epsilon=0.001,
        n_steps=5000,
        m=64,
        restrict_to_orth_grad=True,
    )
    plot_lambda_per_checkpoint("experiment_params/exp2.json", sgld_params)
    # sgld_params = SGLDParams(
    #     gamma=5,
    #     epsilon=0.001,
    #     n_steps=5000,
    #     m=64,
    #     restrict_to_orth_grad=False,
    # )
    # plot_lambda_per_checkpoint("experiment_params/exp2.json", sgld_params)