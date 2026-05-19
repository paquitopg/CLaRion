import torch
import torch.nn.functional as F


def clara_lm_loss(logits: torch.Tensor, labels: torch.Tensor):
    """
    Standard next-token prediction loss
    logits: (B, T, V)
    labels: (B, T)
    """

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )