import math
import torch
from torch import nn

def positions(length, dim, device, dtype):
    index = torch.arange(length, device=device, dtype=dtype)[:, None]
    frequency = torch.exp(torch.arange(0, dim, 2, device=device, dtype=dtype) * (-math.log(10000.0) / dim))
    result = torch.zeros(length, dim, device=device, dtype=dtype)
    result[:, 0::2] = torch.sin(index * frequency)
    result[:, 1::2] = torch.cos(index * frequency[:dim // 2])
    return result

class PromptInjectedSignTransformer(nn.Module):
    """Frozen pre-LN backbone, with trainable per-layer prompt residuals."""
    def __init__(self, input_dim, prompt_dim, hidden_dim, num_layers=12,
                 num_heads=12, prompt_length=20, patch_size=16):
        super().__init__()
        if prompt_dim != hidden_dim or hidden_dim % num_heads:
            raise ValueError("prompt_dim must equal hidden_dim and be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.patch_embed = nn.Conv2d(3, hidden_dim, patch_size, stride=patch_size)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(hidden_dim, num_heads,
            dim_feedforward=4*hidden_dim, batch_first=True, norm_first=True, dropout=0.0) for _ in range(num_layers)])
        for module in (self.patch_embed, self.input_proj, self.layers):
            module.requires_grad_(False)
        self.alpha = nn.Parameter(torch.zeros(num_layers))
        self.prompt_residual = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def embed(self, visual, frame_mask=None):
        if visual.ndim == 5:
            b, t, c, h, w = visual.shape
            x = self.patch_embed(visual.reshape(b*t, c, h, w)).flatten(2).transpose(1, 2)
            patches = x.size(1)
            x = x.reshape(b, t, patches, -1)
            x = x + positions(patches, self.hidden_dim, x.device, x.dtype)[None, None]
        elif visual.ndim == 3:
            b, t, _ = visual.shape
            patches = 1
            x = self.input_proj(visual)[:, :, None]
        else:
            raise ValueError("Expected RGB [B,T,3,H,W] or features [B,T,D]")
        if frame_mask is None:
            frame_mask = torch.zeros(b, t, device=visual.device, dtype=torch.bool)
        if frame_mask.all(1).any():
            raise ValueError("Each video must contain at least one valid frame")
        x = x + positions(t, self.hidden_dim, x.device, x.dtype)[None, :, None]
        return x, frame_mask

    def forward(self, embedded, combined_prompts, frame_mask):
        b, t, patches, d = embedded.shape
        # Segment boundaries are per-video, independent of batch padding.
        lengths = (~frame_mask).sum(1)
        s_count = len(combined_prompts)
        segment_ids = torch.div(torch.arange(t, device=embedded.device)[None] * s_count,
                                lengths[:, None], rounding_mode='floor').clamp_max(s_count-1)
        x = embedded.flatten(1, 2)
        token_mask = frame_mask.repeat_interleave(patches, 1)
        segment_ids = segment_ids.repeat_interleave(patches, 1)
        for k, layer in enumerate(self.layers):
            updates = torch.zeros_like(x)
            for s, prompts in enumerate(combined_prompts):
                # Global video context; each segment takes queries conditioned on its own prompts.
                augmented = torch.cat([prompts, x], 1)
                mask = torch.cat([torch.zeros(b, prompts.size(1), device=x.device, dtype=torch.bool), token_mask], 1)
                normed = layer.norm1(augmented)
                attn = layer.self_attn(normed, normed, normed, key_padding_mask=mask, need_weights=False)[0][:, prompts.size(1):]
                z = x + layer.dropout1(attn) + self.alpha[k] * self.prompt_residual(prompts.mean(1))[:, None]
                z = z + layer.dropout2(layer.linear2(layer.dropout(layer.activation(layer.linear1(layer.norm2(z))))))
                updates = updates + z * (segment_ids == s).unsqueeze(-1)
            x = updates.masked_fill(token_mask[:, :, None], 0)
        return x, token_mask
