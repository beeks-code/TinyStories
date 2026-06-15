import torch
import torch.nn as nn

from .blocks import LayerNormalization, Transformer


class GPT_124(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.token_emd = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf = nn.Sequential(*[Transformer(cfg) for _ in range(cfg["n_layers"])])
        self.final_layer_norm = LayerNormalization(cfg["emb_dim"])
        self.output_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, input_ids):
        _, seq_length = input_ids.shape
        tok_embd = self.token_emd(input_ids)
        pos_embd = self.pos_emb(torch.arange(seq_length, device=input_ids.device))
        x = self.drop_emb(tok_embd + pos_embd)
        x = self.trf(x)
        x = self.final_layer_norm(x)
        logits=self.output_head(x)
        return logits
