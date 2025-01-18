import torch
import torch.nn as nn
from Attention import MultiHeadAttention
from LayerNormalisation import LayerNorm
from FFN import FeedForward

class TransformerV1(nn.Module):
    def __init__(self, gpt_config):
        super().__init__()

        # Multi-head self-attention layer
        self.multi_head_attention = MultiHeadAttention(
            dim_in=gpt_config["embed_dim"],
            dim_out=gpt_config["embed_dim"],
            dropout_p=gpt_config["dropout_rate"],
            num_heads=gpt_config["num_heads"],
            context_size=gpt_config["context_length"],
            qkv_bias=gpt_config["qkv_bias"]
        )

        # Layer normalization layers
        self.layer_norm1 = LayerNorm(
            embedding_dim=gpt_config["embed_dim"]
        )
        self.layer_norm2 = LayerNorm(
            embedding_dim=gpt_config["embed_dim"]
        )

        # Feed-forward network
        self.ffn = FeedForward(
            embed_dim=gpt_config["embed_dim"]
        )

        # Dropout
        self.dropout = nn.Dropout(
            p=gpt_config["dropout_rate"]
        )

    def forward(self, X):
        # Layers from input to first shortcut connection
        SHORCUT1 = X

        # 1st sub-layer: Layer normalization layer
        X = self.layer_norm1(X)

        # 2nd sub-layer: Multi-head self-attention layer
        X = self.multi_head_attention(X)

        # 3rd sub-layer: Dropout
        X = self.dropout(X)

        # Shortcut connection
        X += SHORCUT1

        # Layers from first shortcut connection to output
        SHORCUT2 = X

        # 4th sub-layer: Layer normalization layer
        X = self.layer_norm2(X)

        # 5th sub-layer: Feed-forward network
        X = self.ffn(X)

        # 6th sub-layer: Dropout
        X = self.dropout(X)

        # Shortcut connection
        X += SHORCUT2

        return X