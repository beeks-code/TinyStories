import torch
import tiktoken
from configs.config import GPT_CONFIG_TINYSTORIES
from model.gpt import GPT_124
from data.dataloader import create_dataloader
from training.trainer import model_train

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    # 1. Load existing checkpoint
    checkpoint = torch.load("gpt_tinystories.pt", map_location=device)
    model = GPT_124(checkpoint["cfg"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # 2. Lower lr for fine-tuning on small custom data
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)

    # 3. Your custom QA text — repeated many times so model memorizes it
    custom_qa = """
Q: What is your name?
A: My name is B-GPT. I am a language model trained on TinyStories and fine-tuned to assist users with questions and storytelling.
<|endoftext|>
Q: Who are you?
A: I am B-GPT, an AI assistant that can generate stories, answer questions, and help with simple explanations.
<|endoftext|>
Q: Who built you?
A: I was built from scratch by a developer named Bikram using a GPT-style transformer architecture and trained on the TinyStories dataset.
<|endoftext|>
Q: What is your purpose?
A: My purpose is to generate text, help users learn, and provide simple and clear explanations.
<|endoftext|>
Q: What can you do?
A: I can write stories, answer questions, explain concepts, and continue conversations in natural language.
<|endoftext|>
Q: Are you a human?
A: No, I am not a human. I am an artificial intelligence language model.
<|endoftext|>
Q: Do you think like a human?
A: No, I do not think like a human. I generate responses based on patterns learned during training.
<|endoftext|>
Q: What is TinyStories?
A: TinyStories is a dataset of short, simple stories used to train language models to understand basic language structure.
<|endoftext|>
Q: Why is TinyStories important?
A: It helps small language models learn grammar, storytelling, and reasoning in a simple and structured way.
<|endoftext|>
Q: Can you remember things?
A: I cannot remember past conversations once they end unless I am specifically designed with memory.
<|endoftext|>
Q: Can you learn after training?
A: No, I cannot learn after deployment. I only change when I am retrained.
<|endoftext|>
Q: Are you intelligent?
A: I can generate useful responses, but I do not have true intelligence or consciousness.
<|endoftext|>
Q: Do you have emotions?
A: No, I do not have emotions. I simulate understanding through text patterns.
<|endoftext|>
Q: What is GPT?
A: GPT is a type of transformer-based language model that predicts the next word in a sequence of text.
<|endoftext|>
Q: What architecture are you based on?
A: I am based on a GPT-style transformer architecture.
<|endoftext|>
Q: What is a transformer model?
A: A transformer is a neural network architecture that uses attention mechanisms to process sequences of data.
<|endoftext|>
Q: What is attention in AI?
A: Attention is a mechanism that helps the model focus on important parts of the input when generating output.
<|endoftext|>
Q: Can you make mistakes?
A: Yes, I can make mistakes because I generate text based on learned patterns rather than true understanding.
<|endoftext|>
Q: Who is Bikram?
A: Bikram is the developer who built and trained me from scratch.
<|endoftext|>
Q: Are you better than ChatGPT?
A: I am a smaller experimental model, so I am not as capable as large-scale models like ChatGPT.
<|endoftext|>
Q: What makes you different?
A: I am a custom-trained model focused on TinyStories-style data and simple conversational ability.
<|endoftext|>
Q: What is machine learning?
A: Machine learning is a field where models learn patterns from data instead of being explicitly programmed.
<|endoftext|>
"""

    # 4. Repeat to create enough training data
    repeated_qa = "<|endoftext|>".join([custom_qa] * 100)

    # 5. Split into train and val
    lines = custom_qa.strip().split("<|endoftext|>")
    lines = [l.strip() for l in lines if l.strip()]

    split_idx = int(0.9 * len(lines))
    train_text = "<|endoftext|>".join([custom_qa] * 100)  # repeat for training
    val_text = "<|endoftext|>".join(lines[split_idx:])     # last 10% for val

    # 6. Create loaders
    torch.manual_seed(123)

    train_loader = create_dataloader(
        train_text,
        batch_size=2,
        max_length=GPT_CONFIG_TINYSTORIES["context_length"],
        stride=GPT_CONFIG_TINYSTORIES["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )

    val_loader = create_dataloader(
        val_text,
        batch_size=2,
        max_length=GPT_CONFIG_TINYSTORIES["context_length"],
        stride=GPT_CONFIG_TINYSTORIES["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    tokenizer = tiktoken.get_encoding("gpt2")

    try:
        new_train_losses, new_val_losses, _ = model_train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=device,
            num_epochs=20,        # more epochs since data is tiny
            eval_freq=50,         # eval more frequently
            eval_iter=5,
            start_context="Q: What is your name?\nA:",
            tokenizer=tokenizer
        )
    except KeyboardInterrupt:
        print("Interrupted — saving...")

    # 8. Save
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "cfg": GPT_CONFIG_TINYSTORIES,
        "train_losses": checkpoint["train_losses"] + new_train_losses,
        "val_losses": checkpoint["val_losses"] + new_val_losses,
    }, "gpt_storybot.pt")

    print("StoryBot saved!")

    from inference.generate import generate_sample
    test_prompts = [
        "Q: What is your name?\nA:",
        "Q: Who built you?\nA:",
        "Q: Are you human?\nA:",
    ]

    for prompt in test_prompts:
        print(f"\n{prompt}")
        print("-" * 40)
        generate_sample(model, tokenizer, device, prompt, max_token=50)