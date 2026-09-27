from torch import nn
from torch.nn import functional as F

class MultiObjectiveForgettingLoss(nn.Module):
    """Eq. 42; CE consumes logits and targets shifted by one token."""
    def __init__(self, lambda_forget=1., lambda_align=.3, lambda_route=.1,
                 lambda_prompt=.1, lambda_temp=.1, lambda_gloss=1., pad_id=0):
        super().__init__()
        self.weights = dict(forget=lambda_forget, align=lambda_align, route=lambda_route,
                            prompt=lambda_prompt, temp=lambda_temp)
        self.lambda_gloss, self.pad_id = lambda_gloss, pad_id
    def forward(self, output, text_targets, gloss_targets=None):
        text = F.cross_entropy(output['text_logits'].reshape(-1, output['text_logits'].size(-1)), text_targets.reshape(-1), ignore_index=self.pad_id)
        gloss = text.new_zeros(())
        if self.lambda_gloss:
            if gloss_targets is None or output['gloss_logits'] is None:
                raise ValueError("Gold gloss targets required when lambda_gloss is enabled")
            gloss = F.cross_entropy(output['gloss_logits'].reshape(-1, output['gloss_logits'].size(-1)), gloss_targets.reshape(-1), ignore_index=self.pad_id)
        losses = dict(loss_text=text, loss_gloss=gloss, loss_trans=text+self.lambda_gloss*gloss,
                      loss_forget=output['forgetting_loss'], loss_align=output['alignment_loss'],
                      loss_route=output['routing_loss'], loss_prompt=output['prompt_loss'], loss_temp=output['temporal_loss'])
        losses['total_loss'] = losses['loss_trans'] + sum(weight*losses['loss_'+name] for name,weight in self.weights.items())
        return losses
