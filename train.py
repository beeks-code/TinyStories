import time

import tiktoken
import torch

from configs.config import GPT_CONFIG_TINYSTORIES
from data.dataloader import create_dataloader, get_text_subset
from model.gpt import GPT_124
from training.trainer import model_train


def main():
    train_text = get_text_subset("train", n_stories=20000)
    val_text = get_text_subset("validation", n_stories=2000)

    torch.manual_seed(123)

    train_loader = create_dataloader(
        train_text,
        batch_size=2,
        max_length=GPT_CONFIG_TINYSTORIES["context_length"],
        stride=GPT_CONFIG_TINYSTORIES["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0,
    )

    val_loader = create_dataloader(
        val_text,
        batch_size=2,
        max_length=GPT_CONFIG_TINYSTORIES["context_length"],
        stride=GPT_CONFIG_TINYSTORIES["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT_124(GPT_CONFIG_TINYSTORIES).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=0.1)

    print(f"Using {device}")
    start_time = time.time()

    train_losses, val_losses, tokens_seen = model_train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=10,
        eval_freq=200,
        eval_iter=5,
        start_context="Once upon a time",
        tokenizer=tiktoken.get_encoding("gpt2"),
    )

    execution_time_minutes = (time.time() - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "cfg": GPT_CONFIG_TINYSTORIES,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "tokens_seen": tokens_seen,
        },
        "gpt_tinystories.pt",
    )

    print("Model saved!")
    total_params = sum(p.numel() for p in model.parameters())
    print(total_params)


if __name__ == "__main__":
    main()
