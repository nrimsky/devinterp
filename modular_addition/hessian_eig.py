import torch as t
from torch.autograd import grad
from scipy.sparse.linalg import LinearOperator, eigsh
import numpy as np
from calc_lambda import cross_entropy_loss
from train import ExperimentParams
from dataset import make_dataset, train_test_split
from model import MLP

def get_weight_norm(model):
    return sum((p ** 2).sum() for p in model.parameters() if p.requires_grad)

def hessian_eig(
    params_file: str,
    device="cuda",
    n_top_vectors=500,
    param_extract_fn=None
):
    params = ExperimentParams.load_from_file(params_file)
    dataset = make_dataset(params.p)
    dataset, _ = train_test_split(
        dataset,
        params.train_frac,
        params.random_seed,
    )
    model = MLP(params)
    model = model.to(device)
    param_extract_fn = param_extract_fn or (lambda x: x.parameters())
    num_params = sum(p.numel() for p in param_extract_fn(model))

    X_1 = t.stack([dataset[b][0][0] for b in range(len(dataset))]).to(device)
    X_2 = t.stack([dataset[b][0][1] for b in range(len(dataset))]).to(device)
    Y = t.stack([dataset[b][1] for b in range(len(dataset))]).to(device)

    def compute_loss():
        output = model(X_1, X_2)
        return cross_entropy_loss(output, Y) + (params.weight_decay/2) * get_weight_norm(
            model
        )  # hacky way to add weight norm

    def hessian_vector_product(vector):
        model.zero_grad()
        grad_params = grad(compute_loss(), param_extract_fn(model), create_graph=True)
        flat_grad = t.cat([g.view(-1) for g in grad_params])
        grad_vector_product = t.sum(flat_grad * vector)
        hvp = grad(grad_vector_product, param_extract_fn(model), retain_graph=True)
        return t.cat([g.contiguous().view(-1) for g in hvp])

    def matvec(v):
        v_tensor = t.tensor(v, dtype=t.float32, device=device)
        return hessian_vector_product(v_tensor).cpu().detach().numpy()

    linear_operator = LinearOperator((num_params, num_params), matvec=matvec)
    eigenvalues, _ = eigsh(
        linear_operator,
        k=n_top_vectors,
        tol=0.0001,
        which="LM",
        return_eigenvectors=True,
    )
    return eigenvalues


if __name__ == "__main__":
    hessian_eig("exp_params/large_model/0.95_0.json")
