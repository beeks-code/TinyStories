import tiktoken
import torch

from inference.generate import generate_sample
from model.gpt import GPT_124


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load("gpt_storybot.pt", map_location=device)

    model = GPT_124(checkpoint["cfg"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("Model loaded!")

    tokenizer = tiktoken.get_encoding("gpt2")
    prompts = [
        "who are you ",
        "What are you",
        "How are you"
    ]
    
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        print("-" * 40)
        text=generate_sample(model, tokenizer, device, prompt, max_token=100,temperature=0.8,top_k=40)
        text = text.replace("<|endoftext|>", "")
        print(text)


if __name__ == "__main__":
    main()
