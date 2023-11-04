import torch as t
import random
from math import log, sqrt
from dataclasses import dataclass
from tqdm import tqdm

@dataclass
class SGLDParams:
    gamma: float = 1
    epsilon: float = 0.001
    n_steps: int = 10000
    batch_size: int = 512
    n_multiplier: float = 1
    use_mse: bool = False


def cross_entropy_loss(logits, y_s):
    preds = t.nn.functional.softmax(logits, dim=1)
    return -1 * t.mean(t.log(preds[t.arange(len(preds)), y_s] + 1e-7))


def mse_loss(logits, y_s):
    return t.mean((logits - y_s) ** 2)


def get_full_train_loss(model, dataset, device, use_mse=False):
    X = t.stack([dataset[b][0] for b in range(len(dataset))]).to(device)
    Y = t.stack([dataset[b][1] for b in range(len(dataset))]).to(device)
    out = model(X)
    if use_mse:
        return mse_loss(out, Y)
    return cross_entropy_loss(out, Y)


def sgld(model, sgld_params, dataset, device):
    loss_fn = mse_loss if sgld_params.use_mse else cross_entropy_loss
    model = model.to(device)
    n = len(dataset)
    beta = 1 / log(n * sgld_params.n_multiplier)
    init_loss = get_full_train_loss(model, dataset, device, sgld_params.use_mse)
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

    for _ in tqdm(range(sgld_params.n_steps)):
        batch_idx = random.choices(idx, k=sgld_params.batch_size)
        X = t.stack([dataset[b][0] for b in batch_idx]).to(device)
        Y = t.stack([dataset[b][1] for b in batch_idx]).to(device)
        optimizer.zero_grad()
        out = model(X)
        loss_value = loss_fn(out, Y)
        array_loss.append(loss_value.item())
        w = t.nn.utils.parameters_to_vector(model.parameters())
        elasticity_loss_term = (sgld_params.gamma / 2) * t.sum(((w_0 - w) ** 2))
        loss_term = loss_value * n * beta * sgld_params.n_multiplier
        full_loss = (sgld_params.epsilon / 2) * (elasticity_loss_term + loss_term)
        full_loss.backward()
        optimizer.step()
        with t.no_grad():
            # Add noise to the parameters
            eta = t.randn_like(w, device=device) * sqrt(sgld_params.epsilon)
            new_params = t.nn.utils.parameters_to_vector(model.parameters()) + eta
            t.nn.utils.vector_to_parameters(new_params, model.parameters())
    wbic = sgld_params.n_multiplier * n * sum(array_loss) / len(array_loss)
    lambda_hat = (wbic - n_ln_wstar) / log(sgld_params.n_multiplier * n)
    print(array_loss[::len(array_loss) // 20])
    return lambda_hat
