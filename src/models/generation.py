
import torch
import torch.nn as nn
import torch.nn.functional as F

class CLaRaGenerator(nn.Module):
    """
    Generator CLaRa-style:
    - input = query tokens + retrieved memory tokens
    - output = next-token prediction
    - fully differentiable via STE retriever upstream
    """

    def __init__(self, base_lm: nn.Module, hidden_dim: int):
        super().__init__()
        self.lm = base_lm
        self.hidden_dim = hidden_dim

        # projection si retrieval space ≠ LM space
        self.mem_proj = nn.Linear(hidden_dim, base_lm.config.hidden_size)

    def forward(
        self,
        query_hidden: torch.Tensor,
        retrieved_memory: torch.Tensor,
        labels: torch.Tensor = None,
    ):
        """
        query_hidden: (B, Tq, H)
        retrieved_memory: (B, K, D)
        """

        # 1. proj memory tokens into LM space
        mem = self.mem_proj(retrieved_memory)  # (B, K, H)

        # 2. concat like CLaRa paper
        # [query | memory]
        x = torch.cat([query_hidden, mem], dim=1)

        # 3. forward LM
        out = self.lm(inputs_embeds=x)

        logits = out.logits

        loss = None
        if labels is not None:
            # shift for causal LM
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
            )

        return logits, loss


class CLaRaModel(nn.Module):
    def __init__(self, retriever, encoder, generator):
        super().__init__()
        self.retriever = retriever
        self.encoder = encoder
        self.generator = generator

    def forward(self, query, index, labels=None):

        # 1. encode query → logits similarity space
        q_emb = self.encoder(query)  # (B, D)

        # 2. retrieve (STE inside)
        hard, idx = self.retriever(q_emb)

        # 3. gather memory tokens from index
        memory = index[idx]  # (B, K, D)

        # 4. generator forward
        logits, loss = self.generator(q_emb.unsqueeze(1), memory, labels)

        return logits, loss, idx