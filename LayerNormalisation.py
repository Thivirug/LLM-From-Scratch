import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, embedding_dim, eps=1e-5):
        super().__init__()

        self.eps = eps
        self.scale = nn.Parameter(torch.ones(embedding_dim))
        self.shift = nn.Parameter(torch.zeros(embedding_dim))

    def forward(self, X):
        # calculate mean and variance
        mean = X.mean(dim=-1, keepdim=True)
        var = X.var(dim=-1, keepdim=True, unbiased=False)

        # normalize
        X = (X - mean) / torch.sqrt(var + self.eps)

        # scale and shift
        return self.scale * X + self.shift
    
# # Example
# layer_norm = LayerNorm(4)
# X = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.float32)

# print(layer_norm(X).var(dim=-1, keepdim=True, unbiased=False))
