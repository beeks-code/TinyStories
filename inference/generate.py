import torch


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    return torch.tensor(encoded).unsqueeze(0)


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

def generate(model, ids, max_token, context_size, temperature=0.0, top_k=None, eos_id=50256):

    for _ in range(max_token):
        ids_cond = ids[:, -context_size:]
        with torch.no_grad():
            logits = model(ids_cond)
        logits = logits[:, -1, :]
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        if temperature > 0.0:
            logits = logits / temperature

            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            # Sample from the distribution
            ids_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        else:
            ids_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        if ids_next == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
            break

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
            temperature=temperature,
            top_k=top_k,
            
        )

    decoded_text = token_ids_to_text(token_ids, tokenizer)
    decoded_text=decoded_text.replace("\n", " ")
    model.train()
    return decoded_text
