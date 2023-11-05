import torch as t
from torch import nn
from sgld_simple import sgld, SGLDParams
from typing import List
from math import exp


class SingleParamFunction(nn.Module):
    def __init__(self):
        super().__init__()
        self.parameter = t.nn.Parameter(t.zeros(1))

    def forward(self):
        return self.parameter[0]

class QuadApprox(nn.Module):
    def __init__(self, spectrum: List):
        super().__init__()
        self.parameter = t.nn.Parameter(t.zeros(len(spectrum)))
        self.spectrum = t.tensor(spectrum)

    def forward(self):
        return self.parameter, self.spectrum



# sgld_params = SGLDParams(
#     gamma=1,
#     epsilon=0.0001,
#     n_steps=50000,
#     batch_size=64,
#     n_multiplier=100000,
#     loss_fn=lambda x: 0.001 * x**4 + x**6,
# )

# lambda_hat = sgld(SingleParamFunction(), sgld_params)

# print(f"The estimated effective dimensionality of the model is: {lambda_hat.item()}")


n_multipliers = [10, 100, 1000, 10000, 100000]
results = []

for n_multiplier in n_multipliers:
    sgld_params = SGLDParams(
        gamma=1,
        epsilon=0.0001,
        n_steps=10000,
        batch_size=64,
        n_multiplier=n_multiplier,
        loss_fn=lambda stuff: t.sum(stuff[0]**2 * stuff[1])
    )

    n = 20
    spectrum = [exp(-i/2) for i in range(n)]

    lambda_hat = sgld(QuadApprox(spectrum), sgld_params)

    print(f"The estimated effective dimensionality of the model is: {lambda_hat.item()}")
    results.append(lambda_hat.item())

print(results)
