import math

import matplotlib.pyplot as plt


def perplexity(loss):
    return math.exp(loss)


def plot_losses(train_losses, val_losses, output_path="loss_curve.png"):
    plt.figure(figsize=(10, 6), dpi=150)
    plt.plot(train_losses, linewidth=2.5, label="Training Loss")
    plt.plot(val_losses, linewidth=2.5, label="Validation Loss")
    plt.xlabel("Evaluation Step", fontsize=12)
    plt.ylabel("Cross-Entropy Loss", fontsize=12)
    plt.title(
        "GPT-169M Training on TinyStories\nTraining and Validation Loss",
        fontsize=14,
        pad=15,
    )
    plt.legend(frameon=False, fontsize=11)
    plt.grid(alpha=0.3)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    return output_path
