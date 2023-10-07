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
          self.linear2 = t.nn.Linear(params.hidden_size, params.p, bias=False)
        self.gelu = t.nn.GELU()
        self.vocab_size = params.p
        self.linear1r.weight.data /= params.scale_linear_1_factor
        self.linear1l.weight.data /= params.scale_linear_1_factor

        self.saved_activations = {}
        self.params = params

    def forward(self, a, b):
        x1 = self.embedding(a)
        x2 = self.embedding(b)
        if self.params.linear_1_tied:
          x1 = self.linear1r(x1)
          x2 = self.linear1r(x2)
        else:
          x1 = self.linear1l(x1)
          x2 = self.linear1r(x2)
        x = x1 + x2
        x = self.gelu(x)
        x = self.linear2(x)

        if self.params.save_activations:
          if len(a.shape) == 0:
            a = a.unsqueeze(0)
          self.saved_activations[(a[0], b[0])] = x[0].cpu().clone().detach() # (batch_size, embed_dim)

        if self.tie_unembed:
          x = x @ self.embedding.weight.T
        return x
  