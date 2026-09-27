import torch
from torch import nn
from torch.nn import functional as F

class PromptRoutingMechanism(nn.Module):
    """Video/language-conditioned hard Top-K with straight-through gradients."""
    def __init__(self, prompt_dim, hidden_dim, num_prompts=20, tau=1.0, top_k=5):
        super().__init__()
        if tau <= 0 or not 1 <= top_k <= num_prompts:
            raise ValueError("Invalid routing temperature or Top-K budget")
        self.tau, self.top_k, self.num_prompts = tau, top_k, num_prompts
        self.network = nn.Sequential(nn.Linear(hidden_dim, prompt_dim), nn.ReLU(), nn.Linear(prompt_dim, num_prompts))

    def forward(self, hidden_embedding, prompt_bank):
        # Absent signer slots are excluded before sampling and Top-K.
        logits = self.network(hidden_embedding)[:, :prompt_bank.size(1)]
        probs = F.gumbel_softmax(logits, tau=self.tau, hard=False, dim=-1) if self.training else F.softmax(logits / self.tau, dim=-1)
        indices = probs.topk(min(self.top_k, probs.size(1)), dim=-1).indices
        hard = F.one_hot(indices, probs.size(1)).to(probs.dtype)
        selection = hard + (probs[:, None] - probs[:, None].detach()) if self.training else hard
        active = selection @ prompt_bank
        entropy = -(probs * probs.clamp_min(1e-9).log()).sum(-1).mean()
        return active, entropy, probs, indices
