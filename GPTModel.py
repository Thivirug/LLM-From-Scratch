from Transformers import TransformerV1
from LayerNormalisation import LayerNorm
import torch
import torch.nn as nn

class GPTModelV1(nn.Module):
    def __init__(self, gpt_config):
        super().__init__()

        # vector embedding layer
        self.token_embedding = nn.Embedding(
            num_embeddings=gpt_config["vocab_size"],
            embedding_dim=gpt_config["embed_dim"]
        )

        # positional embedding layer
        self.positional_embedding = nn.Embedding(
            num_embeddings=gpt_config["context_length"],
            embedding_dim=gpt_config["embed_dim"]
        )

        # dropout
        self.dropout = nn.Dropout(
            p=gpt_config["dropout_rate"]
        )

        # transformer layers
        self.transformer_layers = nn.Sequential(
            *[TransformerV1(gpt_config) for _ in range(gpt_config["num_layers"])]
        )

        # final layer normalization layer
        self.final_layer_norm = LayerNorm(
            embedding_dim=gpt_config["embed_dim"]
        )

        # output layer
        self.output_layer = nn.Linear(
            in_features=gpt_config["embed_dim"],
            out_features=gpt_config["vocab_size"],
            bias=False
        )

    def forward(self, token_ids):
        # get vector embeddings
        vector_embeddings = self.token_embedding(token_ids)

        # get positional embeddings
        positional_embeddings = self.positional_embedding(
            torch.arange(
                end=token_ids.shape[1], # context length
                device=token_ids.device
            )
        )

        # get input embeddings
        input_embeddings = vector_embeddings + positional_embeddings

        # apply dropout
        input_embeddings = self.dropout(input_embeddings)

        # apply transformer layers
        transformer_output = self.transformer_layers(input_embeddings)

        # apply final layer normalization
        transformer_output_norm = self.final_layer_norm(transformer_output)

        # get logits
        logits = self.output_layer(transformer_output_norm)

        return logits
