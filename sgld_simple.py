import torch as t
import random
from math import log, sqrt
from dataclasses import dataclass
from tqdm import tqdm
from typing import Callable, Optional

@dataclass
class SGLDParams:
    gamma: float = 1
    epsilon: float = 0.001
    n_steps: int = 10000
    batch_size: int = 512
    n_multiplier: float = 1
    loss_fn: Optional[Callable] = None
    temp_multiplier: float = 1



def get_loss(model, loss_fn, inputs=None, labels=None):
    if inputs is None:
        return loss_fn(model())
    return loss_fn(model(inputs), labels)

def mean(lst):
    return sum(lst) / len(lst)


def sgld(model, sgld_params, inputs=None, labels=None):
    if inputs is None:
        n = 1
        inv_temperature = 1 / sgld_params.temp_multiplier
    else:
        n = len(inputs)
        effective_n = n * sgld_params.n_multiplier
        inv_temperature = (effective_n / log(effective_n)) / sgld_params.temp_multiplier

    init_loss = get_loss(model, sgld_params.loss_fn, inputs, labels)
    optimizer = optimizer = t.optim.SGD(
        model.parameters(),
        weight_decay=0,
        lr=1,
    )
    w_0 = (
        t.nn.utils.parameters_to_vector(model.parameters()).detach().clone()
    )
    array_loss = []
    idx = list(range(n))
    
    for _ in tqdm(range(sgld_params.n_steps)):
        if inputs is not None:
            batch_idx = random.choices(idx, k=sgld_params.batch_size)
            X = t.stack([inputs[b] for b in batch_idx])
            Y = t.stack([labels[b] for b in batch_idx])
        else:
            X = inputs
            Y = labels
        optimizer.zero_grad()
        loss_value = get_loss(model, sgld_params.loss_fn, X, Y)
        array_loss.append(loss_value.item())
        w = t.nn.utils.parameters_to_vector(model.parameters())
        elasticity_loss_term = (sgld_params.gamma / 2) * t.sum(((w_0 - w) ** 2))

        loss_term = loss_value * inv_temperature

        full_loss = (sgld_params.epsilon / 2) * (elasticity_loss_term + loss_term)

        full_loss.backward()
        optimizer.step()
        with t.no_grad():
            # Add noise to the parameters
            eta = t.randn_like(w) * sqrt(sgld_params.epsilon)
            new_params = t.nn.utils.parameters_to_vector(model.parameters()) + eta
            t.nn.utils.vector_to_parameters(new_params, model.parameters())
            start_pos = len(array_loss) // 4

    lambda_hat = (mean(array_loss[start_pos:]) - init_loss) * inv_temperature

    print("Init loss", init_loss.item())
    print("Mean loss", mean(array_loss[start_pos:]))
    print("Inv temp", inv_temperature)
    # For debugging
    print(array_loss[::len(array_loss) // 20])
    return lambda_hat
