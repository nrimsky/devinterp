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


def cross_entropy_loss(logits, y_s):
    """
    logits: outputs of model
    y: target labels

    returns: mean cross entropy loss
    """
    preds = t.nn.functional.softmax(logits, dim=1)
    return -1 * t.mean(t.log(preds[t.arange(len(preds)), y_s] + 1e-6))


def slgd(model, gamma, epsilon, n_steps, m, dataset, beta, device):
    """
    model: MLP model
    gamma: radius
    epsilon: noise std
    n_steps: number of gradient steps / minibatches
    m: minibatch size
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
    for step in tqdm(range(n_steps)):
        batch_idx = random.sample(idx, m)
        X_1 = t.stack([dataset[b][0][0] for b in batch_idx]).to(device)
        X_2 = t.stack([dataset[b][0][1] for b in batch_idx]).to(device)
        Y = t.stack([dataset[b][1] for b in batch_idx]).to(device)
        optimizer.zero_grad()
        out = model(X_1, X_2)
        cross_entropy_loss_value = cross_entropy_loss(out, Y)
        array_loss.append(cross_entropy_loss_value.item())
        w = t.nn.utils.parameters_to_vector(model.parameters())
        elasticity_loss_term = (gamma / 2) * t.sum(((w_star - w) ** 2))
        log_likelihood_loss_term = cross_entropy_loss_value * n * beta
        full_loss = (epsilon / 2) * (elasticity_loss_term + log_likelihood_loss_term)
        full_loss.backward()
        optimizer.step()
        eta = t.randn_like(w, device=device) * sqrt(epsilon)
        with t.no_grad():
            new_params = t.nn.utils.parameters_to_vector(model.parameters()) + eta
            t.nn.utils.vector_to_parameters(new_params, model.parameters())
    wbic = n * sum(array_loss) / len(array_loss)
    lambda_hat = (wbic - n_ln_wstar) / log(n)
    print(f"lambda_hat: {lambda_hat}")
    print(f"wbic: {wbic}")
    print(f"n_ln_wstar: {n_ln_wstar}")
    print(f"init_loss: {init_loss}")
    print(f"array_loss: {array_loss[::len(array_loss)//50]}")
    return model, lambda_hat


def hyperparameter_search(
    params_modular_addition_file, params_random_file, epsilon, n_steps, gamma_range
):
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
        modular_addition_dataset, params_modular_addition.train_frac, params_modular_addition.random_seed
    )
    beta = 1 / log(len(modular_addition_dataset))
    results_random = []
    results_modular_addition = []
    for gamma in gamma_range:
        mlp_modular_addition = MLP(params_modular_addition)
        mlp_random = MLP(params_random)
        mlp_modular_addition.load_state_dict(
            t.load(f"models/model_{params_modular_addition.get_suffix()}.pt")
        )
        mlp_random.load_state_dict(
            t.load(f"models/model_{params_random.get_suffix()}.pt")
        )
        _, lambda_hat_modular_addition = slgd(
            mlp_modular_addition,
            gamma,
            epsilon,
            n_steps,
            params_modular_addition.m,
            modular_addition_dataset,
            beta,
            params_modular_addition.device,
        )
        _, lambda_hat_random = slgd(
            mlp_random,
            gamma,
            epsilon,
            n_steps,
            params_random.m,
            random_dataset,
            beta,
            params_random.device,
        )
        results_modular_addition.append(lambda_hat_modular_addition)
        results_random.append(lambda_hat_random)
    # plot results
    plt.clf()
    plt.figure()
    plt.plot(gamma_range, results_modular_addition, label="modular addition")
    plt.plot(gamma_range, results_random, label="random")
    plt.xlabel("$\gamma$")
    plt.ylabel("$\hat{\lambda}$")
    plt.legend()
    plt.savefig(f'plots/lambda_vs_gamma_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    plt.close()

def get_lambda(param_file):
    params = ExperimentParams.load_from_file(param_file)
    model = MLP(params)
    model.load_state_dict(t.load(f"models/model_{params.get_suffix()}.pt"))
    if params.use_random_dataset:
        dataset = make_random_dataset(params.p, params.random_seed)
    else:
        dataset = make_dataset(params.p)
    train_data, _ = train_test_split(dataset, params.train_frac, params.random_seed)
    gamma = 2
    epsilon = 0.001
    n_steps = 20000
    m = 512
    beta = 1 / log(len(train_data))
    model, lambda_hat = slgd(
        model, gamma, epsilon, n_steps, m, train_data, beta, params.device
    )
    print(lambda_hat)


if __name__ == "__main__":
    # todo: middle layer freezing in both training and measurement
    # generate and test models with different numbers of fourier modes
    # vary p vs lambda
    # compare to random commutative operation
    params_random_file = "models/params_RANDOM_P53_frac0.8_hid32_emb8_tieunembedTrue_tielinFalse_freezeFalse_run6.json"
    params_modular_addition_file = "models/params_P53_frac0.8_hid32_emb8_tieunembedTrue_tielinFalse_freezeFalse_run7.json"
    epsilon = 0.001
    n_steps = 1000
    gamma_range = [1, 2]
    hyperparameter_search(params_modular_addition_file, params_random_file, epsilon, n_steps, gamma_range)
