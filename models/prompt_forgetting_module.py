import torch
from torch import nn
from torch.nn import functional as F

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, scale):
        ctx.scale = scale
        return value.view_as(value)
    @staticmethod
    def backward(ctx, gradient):
        return -ctx.scale * gradient, None

class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_
    def forward(self, value):
        return GradientReversalFunction.apply(value, self.lambda_)

class PromptForgettingModule(nn.Module):
    def __init__(self, prompt_dim, num_signers, beta=.1, lambda_grl=1.0):
        super().__init__()
        self.beta = beta
        self.grl = GradientReversalLayer(lambda_grl)
        self.signer_classifier = nn.Linear(prompt_dim, num_signers)
    def forward(self, pooled_features, language_prompts, signer_prompts, signer_labels):
        logits = self.signer_classifier(self.grl(pooled_features))
        clf = F.cross_entropy(logits, signer_labels)
        decor = logits.new_zeros(())
        if signer_prompts is not None:
            decor = (F.normalize(language_prompts, dim=-1) @ F.normalize(signer_prompts, dim=-1).transpose(1, 2)).square().mean()
        return {'signer_logits': logits, 'forgetting_loss': clf, 'decorrelation_loss': decor,
                'total_forgetting_loss': clf + self.beta * decor}
