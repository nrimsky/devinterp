import torch as t
from tqdm import tqdm
import random
from helpers import eval_model
from math import log, sqrt
from train import ExperimentParams
from model import MLP
from dataset import make_dataset, train_test_split, make_random_dataset
from matplotlib import pyplot as plt
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class SLGDParams:
    gamma: float = 1
    epsilon: float = 0.001
    n_steps: int = 10000
    m: int = 512


def cross_entropy_loss(logits, y_s):
    """
    logits: outputs of model
    y: target labels

    returns: mean cross entropy loss
    """
    preds = t.nn.functional.softmax(logits, dim=1)
    return -1 * t.mean(t.log(preds[t.arange(len(preds)), y_s] + 1e-6))


def slgd(model, slgd_params, dataset, beta, device):
    """
    model: MLP model
    slgd_params: SLGDParams object
    dataset: dataset to train on
    beta: temperature parameter

    returns: updated model, lambda_hat
    """
    n = len(dataset)
    model = model.to(device)
    init_loss = eval_model(model, dataset, device)
    n_ln_wstar = n * init_loss
    idx = list(range(len(dataset)))
    optimizer = t.optim.SGD(model.parameters(), lr=1, weight_decay=0)
    w_star = (
        t.nn.utils.parameters_to_vector(model.parameters()).detach().clone().to(device)
    )
    array_loss = []
    for _ in tqdm(range(slgd_params.n_steps)):
        batch_idx = random.sample(idx, slgd_params.m)
        X_1 = t.stack([dataset[b][0][0] for b in batch_idx]).to(device)
        X_2 = t.stack([dataset[b][0][1] for b in batch_idx]).to(device)
        Y = t.stack([dataset[b][1] for b in batch_idx]).to(device)
        optimizer.zero_grad()
        out = model(X_1, X_2)
        cross_entropy_loss_value = cross_entropy_loss(out, Y)
        array_loss.append(cross_entropy_loss_value.item())
        w = t.nn.utils.parameters_to_vector(model.parameters())
        elasticity_loss_term = (slgd_params.gamma / 2) * t.sum(((w_star - w) ** 2))
        log_likelihood_loss_term = cross_entropy_loss_value * n * beta
        full_loss = (slgd_params.epsilon / 2) * (
            elasticity_loss_term + log_likelihood_loss_term
        )
        full_loss.backward()
        optimizer.step()
        eta = t.randn_like(w, device=device) * sqrt(slgd_params.epsilon)
        with t.no_grad():
            new_params = t.nn.utils.parameters_to_vector(model.parameters()) + eta
            t.nn.utils.vector_to_parameters(new_params, model.parameters())
    wbic = n * sum(array_loss) / len(array_loss)
    lambda_hat = (wbic - n_ln_wstar) / log(n)
    print(f"lambda_hat: {lambda_hat}")
    print(f"wbic: {wbic}")
    print(f"n_ln_wstar: {n_ln_wstar}")
    print(f"init_loss: {init_loss}")
    print(f"slgd_params: {slgd_params}")
    print(f"array_loss: {array_loss[::len(array_loss)//50]}")
    return model, lambda_hat


def hyperparameter_search(
    params_modular_addition_file,
    params_random_file,
    n_steps,
    m,
    epsilon_range,
    gamma_range,
):
    slgd_params = SLGDParams()
    slgd_params.m = m
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
        slgd_params.epsilon = epsilon
        slgd_params.n_steps = actual_n_steps
        for gamma in gamma_range:
            mlp_modular_addition = MLP(params_modular_addition)
            mlp_random = MLP(params_random)
            slgd_params.gamma = gamma
            mlp_modular_addition.load_state_dict(
                t.load(f"models/model_{params_modular_addition.get_suffix()}.pt")
            )
            mlp_random.load_state_dict(
                t.load(f"models/model_{params_random.get_suffix()}.pt")
            )
            _, lambda_hat_modular_addition = slgd(
                mlp_modular_addition,
                slgd_params,
                modular_addition_dataset,
                beta,
                params_modular_addition.device,
            )
            _, lambda_hat_random = slgd(
                mlp_random,
                slgd_params,
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
            f'plots/lambda_vs_gamma_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
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
            f'plots/lambda_vs_epsilon_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        )
        plt.close()


def get_lambda(param_file, slgd_params):
    params = ExperimentParams.load_from_file(param_file)
    model = MLP(params)
    model.load_state_dict(t.load(f"models/model_{params.get_suffix()}.pt"))
    if params.use_random_dataset:
        dataset = make_random_dataset(params.p, params.random_seed)
    else:
        dataset = make_dataset(params.p)
    train_data, _ = train_test_split(dataset, params.train_frac, params.random_seed)
    beta = 1 / log(len(train_data))
    _, lambda_hat = slgd(model, slgd_params, train_data, beta, params.device)
    return lambda_hat


def plot_lambda_per_quantity(param_files, quantity_values, quantity_name, slgd_params):
    lambda_values = []
    for param_file in param_files:
        lambda_values.append(get_lambda(param_file, slgd_params))
    plt.clf()
    plt.figure()
    plt.plot(quantity_values, lambda_values)
    plt.title(f"$\lambda$ vs {quantity_name}")
    plt.xlabel(quantity_name)
    plt.ylabel("$\hat{\lambda}$")
    plt.savefig(
        f'plots/lambda_vs_{quantity_name.replace(" ", "_")}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    )
    plt.close()


if __name__ == "__main__":
    # TODO:
    # Freeze middle layer in both training and lambda measurement sampling
    # Generate and test models with different numbers of Fourier modes
    # Vary P vs \lambda
    hyperparameter_search(
        params_random_file="models/params_RANDOM_P53_frac0.8_hid32_emb8_tieunembedTrue_tielinFalse_freezeFalse_run6.json",
        params_modular_addition_file="models/params_P53_frac0.8_hid32_emb8_tieunembedTrue_tielinFalse_freezeFalse_run7.json",
        n_steps=50,
        m=256,
        epsilon_range=[0.001, 0.01],
        gamma_range=[0.1, 1, 1.5, 2],
    )
