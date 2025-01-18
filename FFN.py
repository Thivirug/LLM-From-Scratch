import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                in_features=embed_dim,
                out_features=embed_dim * 4
            ),

            GELU(),

            nn.Linear(
                in_features=embed_dim * 4,
                out_features=embed_dim
            )
        )

    def forward(self, X):
        return self.network(X)


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return 0.5 * input * (1 + torch.tanh((torch.sqrt(torch.tensor(2 / torch.pi))))) * (input + 0.044715 * torch.pow(input, 3))