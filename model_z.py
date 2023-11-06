import torch as t
from typing import Callable
from sgld_simple import sgld, SGLDParams
from matplotlib import pyplot as plt
from math import exp

class ModelZ(t.nn.Module):

    def __init__(self, act_fn: Callable, hidden_dim: int):
        super().__init__()
        self.act_fn = act_fn
        self.emb_1 = t.nn.Linear(2, hidden_dim, bias=True)
        self.emb_2 = t.nn.Linear(2, hidden_dim, bias=True)
        self.out = t.nn.Linear(hidden_dim, 2, bias=True)

    def forward(self, X) -> t.Tensor:
        x1, x2 = X[:, :2], X[:, 2:]
        x_1 = self.emb_1(x1)
        x_2 = self.emb_2(x2)
        x1 = self.act_fn(x_1 + x_2)
        x1 = self.out(x1)
        return x1
    

def get_dataset(n: int, noise_std: float = 0):
    dataset_x = t.randn((n, 4))
    labels = t.stack([dataset_x[:, 0] * dataset_x[:, 2] - dataset_x[:, 1] * dataset_x[:, 3],
                      dataset_x[:, 0] * dataset_x[:, 3] + dataset_x[:, 1] * dataset_x[:, 2]], dim=1)
    labels += noise_std * t.randn_like(labels)
    return dataset_x, labels
    

def train(act_fn: Callable, hidden_dim: int, epochs: int, lr: float, dataset: t.Tensor, labels: t.Tensor):
    model = ModelZ(act_fn, hidden_dim)
    optim = t.optim.Adam(model.parameters(), lr=lr)
    loss_fn = t.nn.MSELoss()
    for epoch in range(epochs):
        optim.zero_grad()
        pred = model(dataset)
        loss = loss_fn(pred, labels)
        loss.backward()
        optim.step()
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: {loss.item()}")
    return model
        

def experiment():
    n_multipliers = [exp(i) for i in range(-4, 3)]
    noises = [0, 0.1, 0.3]
    full_results = []
    for noise in noises:
        results = []
        for sample in range(5):
            sample_results = []
            for n_multiplier in n_multipliers:
                n = 1000
                n_epochs = 3000
                hidden_dim = 20
                lr = 0.02
                dataset, labels = get_dataset(n, noise)
                act_fn = lambda x: x**2
                loss_fn = t.nn.MSELoss()
                model = train(act_fn, hidden_dim, n_epochs, lr, dataset, labels)
                sgld_params = SGLDParams(
                    gamma=5,
                    epsilon=0.0001,
                    n_steps=5000,
                    batch_size=n,
                    n_multiplier=n_multiplier,
                    loss_fn=loss_fn,
                )
                lambda_hat = sgld(model, sgld_params, dataset, labels)
                print(lambda_hat.item(), n_multiplier, noise)
                sample_results.append(lambda_hat.item())
            results.append(sample_results)
        full_results.append([sum(i) / len(i) for i in zip(*results)])

    with open("results_quad.txt", "w") as f:
        for noise, result in zip(noises, full_results):
            f.write(f"{noise} {result}\n")
        # write the multipliers
        f.write(f"multipliers: {n_multipliers}\n")

    for noise, result in zip(noises, full_results):
        plt.plot(list(range(len(n_multipliers))), result, label=f"noise={noise}", marker="o")

    # x ticks are n_multipliers
    plt.xticks(list(range(len(n_multipliers))), [round(i, 2) for i in n_multipliers])
    plt.legend()
    plt.savefig("results_quad_5_samples.png")

        
if __name__ == "__main__":
    experiment()