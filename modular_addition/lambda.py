import torch as t
from tqdm import tqdm
import random
from helpers import eval_model
from math import log, sqrt
from train import ExperimentParams
from model import MLP
from dataset import make_dataset

def cross_entropy_loss(logits, y_s):
    """
    logits: outputs of model
    y: target labels

    returns: mean cross entropy loss
    """
    preds = t.nn.functional.softmax(logits, dim=1)
    return -1*t.mean(t.log(preds[t.arange(len(preds)), y_s] + 1e-6))

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
    w_star = t.nn.utils.parameters_to_vector(model.parameters()).detach().clone().to(device)
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

if __name__ == "__main__":
    # todo: middle layer freezing in both training and measurement
    # generate and test models with different numbers of fourier modes 
    # vary p vs lambda
    # compare to random commutative operation
    # params = ExperimentParams.load_from_file("models/params_P53_frac0.8_hid32_emb8_tieTrue_freezeFalse_run1.json")
    params = ExperimentParams.load_from_file("models/params_P53_frac0.8_hid32_emb8_tieTrue_freezeFalse.json")
    model = MLP(params)
    model.load_state_dict(t.load(f"models/model_{params.get_suffix()}.pt"))
    dataset = make_dataset(params.p)
    gamma = 2
    epsilon = 0.001
    n_steps = 20000
    m = 512
    beta = 1 / log(len(dataset))
    model, lambda_hat = slgd(model, gamma, epsilon, n_steps, m, dataset, beta, params.device)
    print(lambda_hat)

