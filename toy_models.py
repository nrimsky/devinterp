import torch as t
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim
from sgld_simple import sgld, SGLDParams


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.parameter = t.nn.Parameter(t.zeros(1))

    def forward(self):
        return self.parameter[0]


sgld_params = SGLDParams(
    gamma=1,
    epsilon=0.001,
    n_steps=10000,
    batch_size=64,
    n_multiplier=10000,
    loss_fn=lambda x: x**6 + x**4,
)

lambda_hat = sgld(Model(), sgld_params)

print(f"The estimated effective dimensionality of the model is: {lambda_hat}")
