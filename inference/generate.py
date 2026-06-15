import torch


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    return torch.tensor(encoded).unsqueeze(0)


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

def generate(model, ids, max_token, context_size, temperature=0.0, top_k=None, eos_id=50256):

    # For-loop is the same as before: Get logits, and only focus on last time step
    for _ in range(max_token):
        ids_cond = ids[:, -context_size:]
        with torch.no_grad():
            logits = model(ids_cond)
        logits = logits[:, -1, :]

        # New: Filter logits with top_k sampling
        if top_k is not None:
            # Keep only top_k values
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        # New: Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature

            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            # Sample from the distribution
            ids_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # Otherwise same as before: get ids of the vocab entry with the highest logits value
        else:
            ids_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        if ids_next == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
            break

        # Same as before: append sampled index to the running sequence
        ids = torch.cat((ids, ids_next), dim=1)  # (batch_size, num_tokens+1)

    return ids

# def text_gen(model, ids, max_token, context_size):
#     for _ in range(max_token):
#         ids_cond = ids[:, -context_size:]
#         with torch.no_grad():
#             logits = model(ids_cond)
#         logits = logits[:, -1, :]
#         probs = torch.softmax(logits, dim=-1)
#         next_token = torch.argmax(probs, keepdim=True, dim=-1)
#         ids = torch.cat((ids, next_token), dim=1)
#     return ids


def generate_sample(model, tokenizer, device, start_context, max_token,temperature,top_k):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)

    with torch.no_grad():
        token_ids = generate(
            model=model,
            ids=encoded,
            max_token=max_token,
            context_size=context_size,
            temperature=1.2,
            top_k=5,
            
        )

    decoded_text = token_ids_to_text(token_ids, tokenizer)
    decoded_text=decoded_text.replace("\n", " ")
    model.train()
    return decoded_text
