import torch as t
import random
from math import log, sqrt
from dataclasses import dataclass


@dataclass
class SGLDParams:
    gamma: float = 1
    epsilon: float = 0.001
    n_steps: int = 10000
    m: int = 512
    n_multiplier: float = 1
    weight_decay: float = 0


def cross_entropy_loss(logits, y_s):
    preds = t.nn.functional.softmax(logits, dim=1)
    return -1 * t.mean(t.log(preds[t.arange(len(preds)), y_s] + 1e-7))


def eval_model(model, dataset, device):
    model.eval()
    model.to(device)
    avg_loss = 0
    loss_fn = t.nn.CrossEntropyLoss()
    with t.no_grad():
        for (x1, x2), y in dataset:
            out = model(x1.to(device), x2.to(device)).cpu()
            avg_loss += loss_fn(out, y)
    return avg_loss / len(dataset)


def get_full_train_loss(model, dataset, device):
    X_1 = t.stack([dataset[b][0][0] for b in range(len(dataset))]).to(device)
    X_2 = t.stack([dataset[b][0][1] for b in range(len(dataset))]).to(device)
    Y = t.stack([dataset[b][1] for b in range(len(dataset))]).to(device)
    out = model(X_1, X_2)
    return cross_entropy_loss(out, Y)


def sgld(model, sgld_params, dataset, device):
    model = model.to(device)
    n = len(dataset)
    beta = 1 / log(n * sgld_params.n_multiplier)
    init_loss = get_full_train_loss(model, dataset, device)
    n_ln_wstar = n * init_loss * sgld_params.n_multiplier
    optimizer = optimizer = t.optim.SGD(
        model.parameters(),
        weight_decay=0,
        lr=1,
    )
    w_0 = (
        t.nn.utils.parameters_to_vector(model.parameters()).detach().clone().to(device)
    )
    array_loss = []
    idx = list(range(len(dataset)))

    for _ in range(sgld_params.n_steps):
        batch_idx = random.choices(idx, k=sgld_params.m)
        X_1 = t.stack([dataset[b][0][0] for b in batch_idx]).to(device)
        X_2 = t.stack([dataset[b][0][1] for b in batch_idx]).to(device)
        Y = t.stack([dataset[b][1] for b in batch_idx]).to(device)
        optimizer.zero_grad()
        out = model(X_1, X_2)
        cross_entropy_loss_value = cross_entropy_loss(out, Y)
        array_loss.append(cross_entropy_loss_value.item())
        w = t.nn.utils.parameters_to_vector(model.parameters())
        elasticity_loss_term = (sgld_params.gamma / 2) * t.sum(((w_0 - w) ** 2))
        reg_term = t.sum(w**2) * (sgld_params.weight_decay / 2)
        log_likelihood_loss_term = (
            (cross_entropy_loss_value + reg_term) * n * beta * sgld_params.n_multiplier
        )
        full_loss = (sgld_params.epsilon / 2) * (elasticity_loss_term + log_likelihood_loss_term)
        full_loss.backward()
        optimizer.step()
        with t.no_grad():
            # Add noise to the parameters
            eta = t.randn_like(w, device=device) * sqrt(sgld_params.epsilon)
            new_params = t.nn.utils.parameters_to_vector(model.parameters()) + eta
            t.nn.utils.vector_to_parameters(new_params, model.parameters())
    wbic = sgld_params.n_multiplier * n * sum(array_loss) / len(array_loss)
    lambda_hat = (wbic - n_ln_wstar) / log(sgld_params.n_multiplier * n)
    return lambda_hat
