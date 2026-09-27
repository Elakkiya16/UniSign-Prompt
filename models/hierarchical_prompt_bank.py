import torch
from torch import nn

class HierarchicalCrossLingualPromptBank(nn.Module):
    """Residual language/signer prompt partitions (Eqs. 15–22)."""
    def __init__(self, language_classes, signer_classes, d_model,
                 family_prompt_count=12, signer_prompt_count=8, gamma=0.1):
        super().__init__()
        self.gamma = gamma
        self.language_embed = nn.Embedding(len(language_classes), d_model)
        self.signer_embed = nn.Embedding(max(1, len(signer_classes)), d_model)
        self.language_bank = nn.Parameter(torch.randn(family_prompt_count, d_model) * .02)
        self.signer_bank = nn.Parameter(torch.randn(signer_prompt_count, d_model) * .02)
        self.language_mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.signer_mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model))

    def forward(self, language_ids, signer_ids=None):
        language = self.language_bank[None] + self.language_mlp(self.language_embed(language_ids))[:, None]
        signer = None
        if self.training and signer_ids is not None:
            signer = self.signer_bank[None] + self.signer_mlp(self.signer_embed(signer_ids))[:, None]
        bank = language if signer is None else torch.cat([language, signer], dim=1)
        return bank, language, signer

    def prompt_regularization_loss(self):
        return self.language_bank.square().sum() + self.gamma * self.signer_bank.square().sum()
