import torch as t


class MLP(t.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.embedding = t.nn.Embedding(params.p, params.embed_dim)
        self.linear1r = t.nn.Linear(params.embed_dim, params.hidden_size, bias=True)
        self.linear1l = t.nn.Linear(params.embed_dim, params.hidden_size, bias=True)
        self.tie_unembed = params.tie_unembed
        if params.tie_unembed:
          self.linear2 = t.nn.Linear(params.hidden_size, params.embed_dim, bias=True)
        else:
          self.linear2 = t.nn.Linear(params.hidden_size, params.p, bias=True)
        self.gelu = t.nn.GELU()
        self.vocab_size = params.p

    def forward(self, x1, x2):
        x1 = self.embedding(x1)
        x2 = self.embedding(x2)
        x1 = self.linear1l(x1)
        x2 = self.linear1r(x2)
        x = x1 + x2
        x = self.gelu(x)
        x = self.linear2(x)
        if self.tie_unembed:
          x = x @ self.embedding.weight.T
        return x
  