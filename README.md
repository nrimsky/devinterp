# DevInterp / use of SGLD to estimate learning coefficient

_Produced during [Developmental Interpretability hackathon](https://events.humanitix.com/hackathon-developmental-interpretability) by Nina Panickssery and Dmitry Vaintrob_

- Using SGLD to estimate learning coefficient $\hat{\lambda}$ for a toy model trained on modular addition (approach based on the paper [Quantifying degeneracy in singular models via the learning coefficient](https://arxiv.org/pdf/2308.12108v1.pdf)).
- Visualizing the development of Fourier modes over training, both by looking at their magnitudes in the weights and by looking at the impact of projecting to particular Fourier subspaces on the loss.
- Distinguishing ["Pizza" vs "Clock"](https://browse.arxiv.org/pdf/2306.17844.pdf) circuits by looking at the relationship between the Fourier transform of post-embed activations and pre-unembed activations. Given a large Fourier mode $k$ post-embed, the pre-unembed activations will show a large mode at $2k \text{ mod } p$ for "Pizza" circuits but only at $k$ for "Clock" circuits.

_Related Alignment Forum post_: [Investigating the learning coefficient of modular addition: hackathon project](https://www.alignmentforum.org/posts/4v3hMuKfsGatLXPgt/investigating-the-learning-coefficient-of-modular-addition)

## Codebase organization

- The main code related to the alignment forum post is in the `modular_addition` directory. `scratch` contains some random follow-up experiment stuff which we didn't have time to clean up.
- `modular_addition/train.py` contains the code needed to train our modular addition model
- `modular_addition/calc_lambda.py` contains the code for SGLD-based estimation of the learning coefficient, which runs against model checkpoints produced by `train.py`

### Example usage

Plotting LLC against training steps for a model trained on modular addition:

Train a modular additional model with `p=53` and save 25 training checkpoints along the way:
```python
# In train.py
params = ExperimentParams(
    linear_1_tied=False,
    tie_unembed=False,
    movie=True,
    scale_linear_1_factor=1.0,
    scale_embed=1.0,
    use_random_dataset=False,
    freeze_middle=False,
    n_batches=1500,
    n_save_model_checkpoints=25,
    lr=0.005,
    magnitude=False,
    ablation_fourier=False,
    do_viz_weights_modes=False,
    batch_size=128,
    num_no_weight_decay_steps=0,
    run_id=0,
    activation="gelu",
    hidden_size=48,
    embed_dim=12,
    train_frac=0.95
)
p_sweep_exp([53], params, "example")
```

Estimate and plot the LLC over checkpoints:
```python
# In calc_lambda.py
params = SGLDParams(
    gamma=5,
    epsilon=0.001,
    n_steps=3000,
    m=64,
    restrict_to_orth_grad=True,
    weight_decay=0.0
)
plot_lambda_per_checkpoint("exp_params/example/53_0.json", params)
```

### Grokking modular addition without weight regularization

You don't need to use weight regularization in training, e.g. we have observed grokking with these params:

```python
params = ExperimentParams(
    linear_1_tied=False,
    tie_unembed=False,
    movie=True,
    scale_linear_1_factor=1.0,
    scale_embed=1.0,
    use_random_dataset=False,
    freeze_middle=False,
    n_batches=2000,
    n_save_model_checkpoints=40,
    lr=0.005,
    magnitude=False,
    ablation_fourier=False,
    do_viz_weights_modes=False,
    batch_size=128,
    num_no_weight_decay_steps=0,
    weight_decay=0,  # No weight decay
    run_id=0,
    activation="gelu",
    hidden_size=32,
    embed_dim=16,
    train_frac=0.95
)
```