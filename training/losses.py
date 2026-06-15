import torch
import torch.nn as nn


def calc_loss(logits, target):
    logits_flat = logits.flatten(0, 1)
    target_flat = target.flatten()
    return nn.functional.cross_entropy(logits_flat, target_flat)


def calc_batch_loss(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    return calc_loss(logits, target_batch)


def get_loader_loss(data_loader, model, device, num_batches=None):
    model.eval()

    total_loss = 0.0
    total_batches = len(data_loader) if num_batches is None else num_batches

    if total_batches == 0:
        return 0.0

    with torch.no_grad():
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i >= total_batches:
                break
            loss = calc_batch_loss(input_batch, target_batch, model, device)
            total_loss += loss.item()

    return total_loss / total_batches
