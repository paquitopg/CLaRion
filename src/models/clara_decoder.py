import torch
import torch.nn as nn


class ClaraDecoder(nn.Module):
    """
    Causal decoder conditionné par mémoire retrieval (top-k embeddings)
    """

    def __init__(self, d_model: int, vocab_size: int, n_layers: int = 6, n_heads: int = 8):
        super().__init__()

        self.d_model = d_model

        self.embed = nn.Embedding(vocab_size, d_model)

        self.mem_proj = nn.Linear(d_model, d_model)

        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            batch_first=True,
            norm_first=True,
        )

        self.decoder = nn.TransformerDecoder(layer, num_layers=n_layers)

        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor, memory: torch.Tensor, tgt_mask=None):
        """
        input_ids: (B, T)
        memory: (B, K, D)
        """

        x = self.embed(input_ids)  # (B, T, D)

        memory = self.mem_proj(memory)  # (B, K, D)

        h = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=tgt_mask
        )

        logits = self.lm_head(h)

        return logits