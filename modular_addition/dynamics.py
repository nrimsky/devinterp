import torch as t
from helpers import get_dft_matrix


def ablate_other_modes_fourier_basis(weights, modes, P):
    cos, sin = get_dft_matrix(P)
    c1 = cos @ weights
    s1 = sin @ weights
    mask = t.zeros_like(c1)
    mask[modes] = 1
    mask[0] = 1
    mask[-1 * t.tensor(modes) + P] = 1
    c1 *= mask
    s1 *= mask
    c1 = cos @ c1 / P
    s1 = sin @ s1 / P
    return c1 + s1


def ablate_other_modes_embed_basis(weights, modes, P):
    cos, sin = get_dft_matrix(P)
    c1 = (
        cos @ weights
    )  # (vocab_size x vocab_size) x (vocab_size x embed_dim) = (vocab_size x embed_dim)
    s1 = (
        sin @ weights
    )  # (vocab_size x vocab_size) x (vocab_size x embed_dim) = (vocab_size x embed_dim)
    all_vecs = t.cat(
        (c1[: P // 2 + 1], s1[P // 2 + 1 :]), dim=0
    )  # (vocab_size x embed_dim)
    modes = [0] + modes + [P - mode for mode in modes]  # (k,)
    a = all_vecs[modes]  # (k x embed_dim)
    a_at = a @ a.T  # (k x k)
    a_at_inv = t.linalg.inv(a_at)  # (k x k)
    w = a.T @ a_at_inv @ a  # (embed_dim x embed_dim)
    return weights @ w  # (vocab_size x embed_dim)


def get_magnitude_modes(weights, P):
    cos, sin = get_dft_matrix(P)
    c1, s1 = cos @ weights, sin @ weights
    return t.sqrt(t.sum(c1**2 + s1**2, dim=-1))
