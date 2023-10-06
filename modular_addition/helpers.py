import torch as t

def get_dft_matrix(P):
    cos = t.zeros((P, P))
    sin = t.zeros((P, P))
    for i in range(P):
        for j in range(P):
            theta = t.tensor(2 * t.pi * i * j / P)
            cos[i, j] = t.cos(theta)
            sin[i, j] = t.sin(theta)
    return cos, sin