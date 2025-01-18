import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, dim_in, dim_out, qkv_bias=False):
        super().__init__()

        # initialize the query, key, and value weights
        self.W_q = nn.Linear(
            in_features=dim_in,
            out_features=dim_out,
            bias=qkv_bias
        )

        self.W_k = nn.Linear(
            in_features=dim_in,
            out_features=dim_out,
            bias=qkv_bias
        )

        self.W_v = nn.Linear(
            in_features=dim_in,
            out_features=dim_out,
            bias=qkv_bias
        )

    def forward(self, X):
        # shape of X: (batch_size, sequence_length, dim_in)

        # calculate query, key, and value matrices
        Q = self.W_q(X) # shape: (batch_size, sequence_length, dim_out)
        K = self.W_k(X) # shape: (batch_size, sequence_length, dim_out)
        V = self.W_v(X) # shape: (batch_size, sequence_length, dim_out)

        # calculate the attention scores
        attention_scores = torch.matmul(Q, K.transpose(1, 2)) # shape: (batch_size, sequence_length, sequence_length)

        # scale the attention scores by the sqrt of output dimension of the key
        attention_scores = attention_scores / torch.sqrt(K.shape[-1])

        # calculate the attention weights (normalized attention scores)
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)

        # calculate context vectors
        context_vectors = torch.matmul(attention_weights, V)

        return context_vectors
    
class CausalSelfAttention(nn.Module):
    def __init__(self, dim_in, dim_out, dropout_p, context_size, qkv_bias=False):
        super().__init__()

        # initialize the query, key, and value weights
        self.W_q = nn.Linear(
            in_features=dim_in,
            out_features=dim_out,
            bias=qkv_bias
        )

        self.W_k = nn.Linear(
            in_features=dim_in,
            out_features=dim_out,
            bias=qkv_bias
        )

        self.W_v = nn.Linear(
            in_features=dim_in,
            out_features=dim_out,
            bias=qkv_bias
        )

        # initialize the dropout layer
        self.dropout = nn.Dropout(p=dropout_p)

        # register the buffer for the mask
        self.register_buffer(
            "MASK",
            torch.triu(torch.ones(context_size, context_size), diagonal=1)
        )

    def forward(self, X):
        # shape of X: (batch_size, sequence_length, dim_in)

        # calculate query, key, and value matrices
        Q = self.W_q(X) # shape: (batch_size, sequence_length, dim_out)
        K = self.W_k(X)
        V = self.W_v(X)

        # calculate the attention scores
        attention_scores = torch.matmul(Q, K.transpose(1, 2))

        # mask the attention scores
        seq_length = X.shape[1]
        attention_scores = attention_scores.masked_fill(self.MASK.bool()[:seq_length, :seq_length], -torch.inf)

        # scale the attention scores by the sqrt of output dimension of the key
        attention_scores = attention_scores / torch.sqrt(K.shape[-1])

        # calculate the attention weights (normalized attention scores)
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)

        # apply dropout to the attention weights
        attention_weights = self.dropout(attention_weights)

        # calculate context vectors
        context_vectors = torch.matmul(attention_weights, V)

        return context_vectors
    
class MultiHeadAttention(nn.Module):
    def __init__(self, dim_in, dim_out, dropout_p, num_heads, context_size, qkv_bias=False):
        super().__init__()

        # assert that dim_out is divisible by num_heads
        assert (dim_out % num_heads) == 0, "dim_out must be divisible by num_heads"

        # initialize the query, key, and value weights 
        self.W_q = nn.Linear(
            in_features=dim_in,
            out_features=dim_out,
            bias=qkv_bias
        )

        self.W_k = nn.Linear(
            in_features=dim_in,
            out_features=dim_out,
            bias=qkv_bias
        )

        self.W_v = nn.Linear(
            in_features=dim_in,
            out_features=dim_out,
            bias=qkv_bias
        )

        # calculate the dimension of each head
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.dim_head = dim_out // num_heads

        # initialize the dropout layer
        self.dropout = nn.Dropout(p=dropout_p)

        # register the buffer for the mask
        self.register_buffer(
            "MASK",
            torch.triu(torch.ones(context_size, context_size), diagonal=1)
        )

        # create the output linear layer
        self.W_o = nn.Linear(
            in_features=dim_out,
            out_features=dim_out
        )

    def forward(self, X):
        # shape of X: (batch_size, sequence_length, dim_in)
        batch_size, seq_length, _ = X.shape

        # calculate query, key, and value matrices
        # shape: (batch_size, sequence_length, dim_out)
        Q = self.W_q(X)
        K = self.W_k(X)
        V = self.W_v(X)

        # unroll last dimension into num_heads & dim_head
        # shape: (batch_size, sequence_length, num_heads, dim_head)
        Q = Q.view(batch_size, seq_length, self.num_heads, self.dim_head)
        K = K.view(batch_size, seq_length, self.num_heads, self.dim_head)
        V = V.view(batch_size, seq_length, self.num_heads, self.dim_head)

        # group by the number of heads
        # shape: (batch_size, num_heads, sequence_length, dim_head)
        Q = Q.transpose(1,2)
        K = K.transpose(1,2)
        V = V.transpose(1,2)

        # calculate the attention scores
        # shape: (batch_size, num_heads, sequence_length, sequence_length)
        attention_scores = torch.matmul(Q, K.transpose(2,3))

        # mask the attention scores
        attention_scores = attention_scores.masked_fill(
            self.MASK.bool()[:seq_length, :seq_length], # mask for the first seq_length elements (For shorter sequences)
            -torch.inf
        )

        # scale the attention scores by the sqrt of head dimension
        attention_scores = attention_scores / torch.sqrt(torch.tensor(self.dim_head))

        # calculate the attention weights (normalized attention scores)
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)

        # apply dropout to the attention weights
        attention_weights = self.dropout(attention_weights)

        # calculate context vectors
        # shape: (batch_size, num_heads, sequence_length, dim_head)
        context_vectors = torch.matmul(attention_weights, V)

        # 1) Reshape the context vectors (Take num_heads and dim_head  close to each other)
        # shape: (batch_size, sequence_length, num_heads, dim_head)
        context_vectors = context_vectors.transpose(1,2)

        # 2) Reshape the context vectors (Combine num_heads and dim_head into dim_out [Back to original shape])
        # shape: (batch_size, sequence_length, dim_out)
        context_vectors = context_vectors.contiguous() # contiguous() is used to make sure that the tensor is stored in a contiguous chunk of memory
        context_vectors = context_vectors.view(batch_size, seq_length, self.dim_out)

        # apply the output linear layer
        context_vectors = self.W_o(context_vectors)

        return context_vectors