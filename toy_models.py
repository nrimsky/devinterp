import torch as t
from torch import nn
from sgld_simple import sgld, SGLDParams
from typing import List
from math import exp
from matplotlib import pyplot as plt
import math

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

class TwoParamFunction(nn.Module):
    def __init__(self):
        super().__init__()
        self.parameter = t.nn.Parameter(t.zeros(2))

    def forward(self):
        return self.parameter[0], self.parameter[1]


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


# temp_multipliers = [30, 3, .3, .03]
# spectra = [
#     [1, 0.1, 0.01, 0.001],
#     [1, 0.1, 0.01, 0],
#     [1, 0.1, 0, 0],
#     [1, 0, 0, 0],
# ]


# def loss_fn(p_s):
#     params, spectrum = p_s
#     return t.sum(params**2 * spectrum) + t.sum(params**4)

# sgld_params = SGLDParams(
#     gamma=0,
#     epsilon=0.01,
#     n_steps=30000,
#     batch_size=64,
#     loss_fn=loss_fn,
# )
# for temp_multiplier in temp_multipliers:
#     sgld_params.temp_multiplier = temp_multiplier
#     lambdas = []
#     models = [QuadApprox(spectrum) for spectrum in spectra]
#     for idx, model in enumerate(models):
#         lambda_hat = sgld(model, sgld_params)
#         print(f"Rank: {4-idx} | T: {temp_multiplier} | Lambda: {lambda_hat.item()}")
#         lambdas.append(lambda_hat.item())
#     plt.plot([4, 3, 2, 1], lambdas, label=f"Temperature: {temp_multiplier}", marker="o", linestyle="--")
# plt.legend()
# plt.xlabel("Hessian Rank (with decaying eigenvalues)")
# plt.ylabel("Estimated $\hat{\lambda}$")
# plt.title("Estimated $\hat{\lambda}$ vs. Hessian Rank at different temperatures")
# plt.savefig("quad_approx_t_scale.png")



temp_multipliers = [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
eps = 0.01
sgld_params = SGLDParams(
    gamma=.05,
    epsilon=0.001,
    n_steps=20000,
    batch_size=64,
    loss_fn=lambda params: (params[0]**2 + eps) * (params[1] ** 2 + eps),
)

lambdas = []
for temp_multiplier in temp_multipliers:
    sgld_params.temp_multiplier = temp_multiplier
    sgld_params.epsilon = 0.01* math.sqrt(temp_multiplier)
    sgld_params.n_steps= int(2000/math.sqrt(temp_multiplier))
    lambda_hat_avg = 0
    for trial in range(5):
        model = TwoParamFunction()
        lambda_hat = sgld(model, sgld_params)
        lambda_hat_avg += lambda_hat.item()
    lambdas.append(lambda_hat_avg / 5)
    
plt.plot(list(range(len(temp_multipliers))), lambdas, marker="o", linestyle="--")
plt.xticks(list(range(len(temp_multipliers))), temp_multipliers)
plt.xlabel("Temperature")
plt.ylabel("Estimated $\hat{\lambda}$")
plt.title("Estimated $\hat{\lambda}$ for $(x^2 + \epsilon)(y^2 + \epsilon)$ vs. Temperature")
plt.savefig("eps_lambda.png")