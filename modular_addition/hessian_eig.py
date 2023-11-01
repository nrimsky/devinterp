import torch as t
from torch.autograd import grad
from scipy.sparse.linalg import LinearOperator, eigsh
from calc_lambda import cross_entropy_loss
from train import ExperimentParams
from dataset import make_dataset, train_test_split
from model import MLP
from matplotlib import pyplot as plt
from datetime import datetime
from typing import Callable, List

def get_weight_norm(model):
    return sum((p ** 2).sum() for p in model.parameters() if p.requires_grad)

def hessian_eig(
    params_file: str,
    device="cuda",
    n_top_vectors=300,
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
        which="LA",
        return_eigenvectors=True,
    )
    return eigenvalues[::-1]


def hessian_eig_sweep(filenames, labels, ntop):
    # Get the rainbow colormap
    cmap = plt.cm.get_cmap('rainbow')
    n_files = len(filenames)
    for i, f in enumerate(filenames):
        eigenvalues = hessian_eig(f, n_top_vectors=ntop) 
        eigenvalues = sorted([float(e) for e in eigenvalues], reverse=True)
        # Get a color from the rainbow colormap based on the file's index
        color = cmap(i / (n_files - 1))
        # plot with no connecting lines
        plt.plot(eigenvalues, "o", color=color, label=labels[i], markersize=2, linestyle="None")
    plt.legend()
    plt.xlabel("Rank")
    plt.ylabel("Eigenvalue")
    plt.savefig(f"plots/eig/eigenvalues_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")


if __name__ == "__main__":
    files = [
        "exp_params/frac_sweep/0.1_0.json",
        "exp_params/frac_sweep/0.15_0.json",
        "exp_params/frac_sweep/0.2_0.json",
        "exp_params/frac_sweep/0.25_0.json",
        "exp_params/frac_sweep/0.3_0.json",
        "exp_params/frac_sweep/0.35_0.json",
        "exp_params/frac_sweep/0.4_0.json",
        "exp_params/frac_sweep/0.45_0.json",
        "exp_params/frac_sweep/0.5_0.json",
        "exp_params/frac_sweep/0.55_0.json",
        "exp_params/frac_sweep/0.6_0.json",
        "exp_params/frac_sweep/0.65_0.json",
        "exp_params/frac_sweep/0.7_0.json",
        "exp_params/frac_sweep/0.75_0.json",
        "exp_params/frac_sweep/0.8_0.json",
        "exp_params/frac_sweep/0.85_0.json",
        "exp_params/frac_sweep/0.9_0.json",
        "exp_params/frac_sweep/0.95_0.json",
    ]
    labels = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 
         0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 
         0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    hessian_eig_sweep(files, labels, 100)


