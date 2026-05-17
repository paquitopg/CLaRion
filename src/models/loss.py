import torch
import torch.nn.functional as F


class ClaraSFTLoss(torch.nn.Module):
    """
    Close to OpenRLHF SFTLoss but adapted for retrieved memory tokens.
    """

    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        """
        logits: (B, T, V)
        labels: (B, T)
        """

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=self.ignore_index,
        )

        return loss