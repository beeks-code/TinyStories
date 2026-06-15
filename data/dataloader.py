import tiktoken
from datasets import load_dataset
from torch.utils.data import DataLoader

from .dataset import GPT_dataset


def create_dataloader(
    txt,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0,
):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPT_dataset(txt, tokenizer, max_length, stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )


def load_tinystories():
    dataset = load_dataset("roneneldan/TinyStories")
    train_text = "<|endoftext|>".join(dataset["train"]["text"])
    val_text = "<|endoftext|>".join(dataset["validation"]["text"])
    return train_text, val_text


def get_text_subset(split, n_stories=50000):
    dataset = load_dataset("roneneldan/TinyStories", streaming=True)
    stories = []

    for i, example in enumerate(dataset[split]):
        if i >= n_stories:
            break
        stories.append(example["text"])

    return "<|endoftext|>".join(stories)
