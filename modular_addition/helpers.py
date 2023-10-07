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


def eval_model(model, dataset, device):
    model.eval()
    avg_loss = 0
    loss_fn = t.nn.CrossEntropyLoss()
    with t.no_grad():
        for (x1, x2), y in dataset:
            out = model(x1.to(device), x2.to(device)).cpu()
            avg_loss += loss_fn(out, y)
    return avg_loss / len(dataset)
