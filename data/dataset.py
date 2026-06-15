import torch
from torch.utils.data import Dataset


class GPT_dataset(Dataset):
    def __init__(self, text, tokenizer, context_size, strides):
        self.input_ids = []
        self.target_ids = []

        tokens = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        for i in range(0, len(tokens) - context_size, strides):
            input_tokens = tokens[i : i + context_size]
            target_tokens = tokens[i + 1 : i + context_size + 1]
            self.input_ids.append(torch.tensor(input_tokens))
            self.target_ids.append(torch.tensor(target_tokens))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]
