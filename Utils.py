import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

def train_test_split(data, train_ratio=0.8):
    train_size = int(len(data) * train_ratio)
    train_data = data[:train_size]
    test_data = data[train_size:]
    return train_data, test_data

def get_device():
    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


# simple func
def generate_text_simple(model, input_tokens, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        input_tokens = input_tokens[:, -context_size:] 

        # Get the model's next token predictions
        with torch.no_grad():
            logits = model(input_tokens) # shape: [batch_size, context_size, vocab_size]

        # get the last token's logits (logits for the next token)
        next_token_logits = logits[:, -1, :] # shape: [batch_size, vocab_size]

        # apply softmax to convert logits to probabilities
        next_token_probs = torch.softmax(next_token_logits, dim=-1)

        # get the token id with the highest probability
        next_token_ids = torch.argmax(next_token_probs, dim=-1, keepdim=True)

        # append the new token to the input tokens
        input_tokens = torch.cat((input_tokens, next_token_ids), dim=-1)

    return input_tokens

# func with temp scaling and topk sampling
def generate_text(model, input_tokens, max_new_tokens, context_size, top_k=None, temperature=0, eos_id=None):
    for _ in range(max_new_tokens):
        # get the last 'context size' tokens 
        input_tokens = input_tokens[:, -context_size:]

        # get predictions
        with torch.no_grad():
            logits = model(input_tokens) # shape: (batch_size, n_tokens, vocab_size)

        # get the last token's logits (logits for the next token)
        logits = logits[:, -1, :] # shape: (batch_size,vocab_size)

        # top-K sampling
        if top_k:
            topK_logits, _ = torch.topk(logits, top_k, dim=-1) #shape: (batch_size, top_k)

            logits = torch.where(
                condition = logits < topK_logits[:,-1],
                #input = torch.tensor(-float("inf")), # creates a tensor on CPU
                input = torch.full_like(logits, -float("inf")), # creates the tensor on whatever device logits is in (CPU or GPU)
                other = logits
            ) # shape: (batch_size, vocab_size)

        # temperature scaling
        if temperature > 0:
            logits = logits / temperature # shape: (batch_size, vocab_size)

            # apply softmax
            probs = torch.nn.functional.softmax(logits, dim=-1) # shape: (batch_size, vocab_size)

            # sample next token id using multinomial distribution
            next_token_id = torch.multinomial(probs, num_samples=1) # shape: (batch_size, 1)

        else:
            # get the token with highest prob using argmax
            next_token_id = torch.argmax(probs, dim=-1, keepdim=True) # shape: (batch_size, 1)

        # stop if next token id = eos id
        if eos_id is not None and (next_token_id == eos_id).any():
            break

        # concat input and predicted tokens
        input_tokens = torch.cat(
            (input_tokens, next_token_id),
            dim=-1
        ) # shape : (batch_size, n_tokens+1)

    return input_tokens

def text_to_tokens(text, tokenizer):
    tokens = tokenizer.tokenize(text)
    tokens = torch.tensor(tokens).unsqueeze(0) # add batch dimension
    return tokens

def tokens_to_text(tokens, tokenizer):
    text = tokenizer.detokenize(tokens.squeeze(0).tolist()) # remove batch dimension and convert to list
    return text

def calc_loss_per_batch(model, device, input_batch_ids, target_batch_ids):
    # move data to device
    input_batch_ids = input_batch_ids.to(device)
    target_batch_ids = target_batch_ids.to(device)

    # get the model's predictions
    logits = model(input_batch_ids) # shape: [batch_size, context_size, vocab_size]

    # calculate the loss
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0,1), # shape: [batch_size*context_size, vocab
        target_batch_ids.flatten() # shape: [batch_size*context_size]
    )

    return loss

def calc_total_loss(model, device, data_loader, n_batches=None):
    total_loss = 0

    if len(data_loader) == 0:
        return total_loss
    
    elif n_batches is None:
        n_batches = len(data_loader)

    else: #if num_batches is specified and greater than the length of the data loader (actual number of batches)
        n_batches = min(n_batches, len(data_loader))

    # iterate over the data loader 
    for i, (input_batch_ids, target_batch_ids) in enumerate(data_loader):
        if i < n_batches:
            # calculate the loss for the current batch
            loss = calc_loss_per_batch(model, device, input_batch_ids, target_batch_ids)

            # add the loss to the total loss
            total_loss += loss.item()

        else:
            break

    return total_loss / n_batches # return the average loss

def eval_model(model, train_text_loader, val_text_loader, device, eval_iter):
    # move the model to the device
    model.to(device)

    # calculate the loss on the training and validation data
    with torch.no_grad():
        train_loss = calc_total_loss(model, device, train_text_loader, n_batches=eval_iter)
        val_loss = calc_total_loss(model, device, val_text_loader, n_batches=eval_iter)

    model.train() # put model back in training mode

    return train_loss, val_loss

# simple func
def gen_print_sample_txt(model, tokenizer, start_context_text):
    model.eval() # put model in evaluation mode

    context_size = model.positional_embedding.weight.shape[0]
    tokens = text_to_tokens(start_context_text, tokenizer)

    # generate new tokens
    with torch.no_grad():
        new_tokens = generate_text_simple(model, tokens, max_new_tokens=100, context_size=context_size)

    # decode the new tokens
    new_text = tokens_to_text(new_tokens, tokenizer)
    print("\n\tNew Text: ", new_text.replace("\n", " "))

    model.train() # put model back in training mode

# better func
def gen_print_sample_txt_V2(model, tokenizer, start_context_text, top_k, temperature, eos_id, device, max_new_tokens):
    # model in eval mode
    model.eval()

    # get tokens
    tokens = text_to_tokens(start_context_text, tokenizer).to(device)

    # generate new tokens
    with torch.no_grad():
        new_tokens = generate_text(
            model=model, 
            input_tokens=tokens, 
            max_new_tokens=max_new_tokens,
            context_size=model.positional_embedding.weight.shape[0],
            top_k=top_k,
            temperature=temperature,
            eos_id=eos_id
        )

    # decode the new tokens
    new_text = tokens_to_text(new_tokens, tokenizer)
    print("\n\tNew Text: ", new_text.replace("\n", " "))
    
    model.train()
    
def train(model, optimizer, device, n_epochs, tokenizer, train_text_loader, val_text_loader, eval_freq, eval_iter, start_context_text):
    tokens_seen, global_steps = 0, -1
    train_losses, val_losses, track_tokens_seen = [], [], []

    for epoch in range(n_epochs):
        # put model in training mode
        model.train()

        for input_batch_ids, target_batch_ids in train_text_loader:
            # reset the gradients
            optimizer.zero_grad()

            # calculate the loss
            loss = calc_loss_per_batch(model, device, input_batch_ids, target_batch_ids)

            # backpropagate
            loss.backward()

            # update the weights
            optimizer.step()

            # update the number of tokens seen and global steps
            tokens_seen += input_batch_ids.numel()
            global_steps += 1

            # evaluate the model for eval_freq times
            if global_steps % eval_freq == 0:
                train_loss, val_loss = eval_model(model, train_text_loader, val_text_loader, device, eval_iter=eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)

                print(f"\nEpoch: {epoch+1} (Step: {global_steps}), Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # generate and print a sample text
        gen_print_sample_txt(model, tokenizer, device, start_context_text)

    return train_losses, val_losses, track_tokens_seen

def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # only show integer labels on x-axis

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room
    plt.savefig("loss-plot.pdf")
    plt.show()

def assign_params(left, right):
    # check if left, right tensors have the same shape
    # & returns the right tensor as a trainable Parameter
    if left.shape != right.shape:
        raise ValueError(f"Shape Mismatch!. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))

def load_weights_into_model(model, params):
    # loading token embedding weights
    model.token_embedding.weight = assign_params(model.token_embedding.weight, params["wte"])

    # loading positional embedding weights
    model.positional_embedding.weight = assign_params(model.positional_embedding.weight, params["wpe"])

    # loading transformer block weights in each transformer
    for b in range(len(params["blocks"])):
        # 1) Attention layer
        # getting query, key, value weights 
        w_q, w_k, w_v = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"],
            3,
            axis=-1
        )

        # setting query, key, value weights to model
        model.transformer_layers[b].multi_head_attention.W_q.weight = assign_params(
            model.transformer_layers[b].multi_head_attention.W_q.weight,
            w_q.T
        )

        model.transformer_layers[b].multi_head_attention.W_k.weight = assign_params(
            model.transformer_layers[b].multi_head_attention.W_k.weight,
            w_k.T
        )

        model.transformer_layers[b].multi_head_attention.W_v.weight = assign_params(
            model.transformer_layers[b].multi_head_attention.W_v.weight,
            w_v.T
        )

        # getting query, key, value biases
        b_q, b_k, b_v = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"],
            3,
            axis=-1
        )

        # setting query, key, value biases to model
        model.transformer_layers[b].multi_head_attention.W_q.bias = assign_params(
            model.transformer_layers[b].multi_head_attention.W_q.bias,
            b_q
        )

        model.transformer_layers[b].multi_head_attention.W_k.bias = assign_params(
            model.transformer_layers[b].multi_head_attention.W_k.bias,
            b_k
        )

        model.transformer_layers[b].multi_head_attention.W_v.bias = assign_params(
            model.transformer_layers[b].multi_head_attention.W_v.bias,
            b_v
        )

        # setting output layer weights and biases
        model.transformer_layers[b].multi_head_attention.W_o.weight = assign_params(
            model.transformer_layers[b].multi_head_attention.W_o.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T
        )
        model.transformer_layers[b].multi_head_attention.W_o.bias = assign_params(
            model.transformer_layers[b].multi_head_attention.W_o.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"]
        )

        # 2) FFN layer
        #layer 1
        model.transformer_layers[b].ffn.network[0].weight = assign_params(
            model.transformer_layers[b].ffn.network[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T
        )
        model.transformer_layers[b].ffn.network[0].bias = assign_params(
            model.transformer_layers[b].ffn.network[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"]
        )

        #layer 2 = GELU (no weights)

        #layer 3
        model.transformer_layers[b].ffn.network[2].weight = assign_params(
            model.transformer_layers[b].ffn.network[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T
        )
        model.transformer_layers[b].ffn.network[2].bias = assign_params(
            model.transformer_layers[b].ffn.network[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"]
        )

        # 3) Layer Norm layers
        # layer 1
        model.transformer_layers[b].layer_norm1.scale = assign_params(
            model.transformer_layers[b].layer_norm1.scale,
            params["blocks"][b]["ln_1"]["g"]
        )
        model.transformer_layers[b].layer_norm1.shift = assign_params(
            model.transformer_layers[b].layer_norm1.shift,
            params["blocks"][b]["ln_1"]["b"]
        )

        # layer 2
        model.transformer_layers[b].layer_norm2.scale = assign_params(
            model.transformer_layers[b].layer_norm2.scale,
            params["blocks"][b]["ln_2"]["g"]
        )
        model.transformer_layers[b].layer_norm2.shift = assign_params(
            model.transformer_layers[b].layer_norm2.shift,
            params["blocks"][b]["ln_2"]["b"]
        )

    # loading final layer norm weights
    model.final_layer_norm.scale = assign_params(
        model.final_layer_norm.scale,
        params["g"]
    )
    model.final_layer_norm.shift = assign_params(
        model.final_layer_norm.shift,
        params["b"]
    )

    # loading output layer weights
    model.output_layer.weight = assign_params(
        model.output_layer.weight,
        params["wte"]
    )