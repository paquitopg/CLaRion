import torch

from .pipeline import search_differentiable
from .decoder_clara import ClaraDecoder
from .loss_clara_sft import ClaraSFTLoss


class ClaraTrainer:

    def __init__(self, bank, topk_module, config):
        self.bank = bank
        self.topk = topk_module

        self.decoder = ClaraDecoder(
            hidden_dim=config.hidden_dim,
            vocab_size=config.vocab_size
        )

        self.loss_fn = ClaraSFTLoss()
        self.opt = torch.optim.AdamW(self.decoder.parameters(), lr=2e-4)

    def train_step(self, queries, labels):

        self.opt.zero_grad()

        # retrieval differentiable (CRITICAL PATH)
        M_k, _, _ = search_differentiable(
            self.bank,
            queries,
            self.topk
        )

        # decode
        logits = self.decoder(M_k)

        # SFT loss (OpenRLHF style)
        loss = self.loss_fn(logits, labels)

        loss.backward()
        self.opt.step()

        return loss.item()