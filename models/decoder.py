import torch
from torch import nn
from .prompt_injected_sign_transformer import positions

class AutoregressiveBranch(nn.Module):
    def __init__(self, dim, vocabulary, heads, layers, pad_id):
        super().__init__()
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocabulary, dim, padding_idx=pad_id)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(dim, heads,
            dim_feedforward=4*dim, dropout=.1, batch_first=True), layers)
        self.output = nn.Linear(dim, vocabulary)
    def forward(self, tokens, memory, memory_mask=None):
        x = self.embedding(tokens)
        x = x + positions(x.size(1), x.size(2), x.device, x.dtype)[None]
        causal = torch.ones(x.size(1), x.size(1), device=x.device, dtype=torch.bool).triu(1)
        hidden = self.decoder(x, memory, tgt_mask=causal,
            tgt_key_padding_mask=tokens.eq(self.pad_id), memory_key_padding_mask=memory_mask)
        return self.output(hidden)

    @torch.no_grad()
    def generate(self, memory, memory_mask, bos_id, eos_id, max_length, beam_size):
        results = []
        for sample in range(memory.size(0)):
            mem, mask = memory[sample:sample+1], memory_mask[sample:sample+1]
            beams = [([bos_id], 0.0)]
            for _ in range(max_length):
                candidates = []
                for sequence, score in beams:
                    if sequence[-1] == eos_id:
                        candidates.append((sequence, score))
                        continue
                    tokens = torch.tensor([sequence], device=mem.device)
                    probabilities = self(tokens, mem, mask)[0, -1].log_softmax(-1)
                    probabilities[self.pad_id] = -torch.inf
                    probabilities[bos_id] = -torch.inf
                    values, indices = probabilities.topk(min(beam_size, probabilities.numel()-2))
                    candidates.extend((sequence + [int(i)], score + float(v)) for v, i in zip(values, indices))
                beams = sorted(candidates, key=lambda pair: pair[1], reverse=True)[:beam_size]
                if all(seq[-1] == eos_id for seq, _ in beams):
                    break
            results.append(beams[0][0][1:])
        output = torch.full((len(results), max(map(len, results))), self.pad_id, device=memory.device, dtype=torch.long)
        for i, result in enumerate(results):
            output[i, :len(result)] = torch.tensor(result, device=memory.device)
        return output

class GlossTextDecoder(nn.Module):
    """Causal token decoders; predicted gloss distributions condition text."""
    def __init__(self, feature_dim, gloss_vocab_size, text_vocab_size,
                 num_heads=12, num_layers=2, pad_id=0, use_gloss=True):
        super().__init__()
        self.use_gloss, self.pad_id = use_gloss, pad_id
        self.text_branch = AutoregressiveBranch(feature_dim, text_vocab_size, num_heads, num_layers, pad_id)
        self.gloss_branch = AutoregressiveBranch(feature_dim, gloss_vocab_size, num_heads, num_layers, pad_id) if use_gloss else None

    def forward(self, memory, gloss_targets=None, text_targets=None, memory_mask=None):
        gloss_logits = None
        if self.use_gloss:
            if gloss_targets is None:
                raise ValueError("Gloss decoder inputs are required for supervised forward; use generate for inference")
            gloss_logits = self.gloss_branch(gloss_targets, memory, memory_mask)
            auxiliary = gloss_logits.softmax(-1) @ self.gloss_branch.embedding.weight
            memory = torch.cat([memory, auxiliary], 1)
            memory_mask = torch.cat([memory_mask, gloss_targets.eq(self.pad_id)], 1)
        if text_targets is None:
            raise ValueError("Shifted text inputs required; use generate for inference")
        return gloss_logits, self.text_branch(text_targets, memory, memory_mask)

    @torch.no_grad()
    def generate(self, memory, memory_mask, bos_id=1, eos_id=2, max_length=64, beam_size=5):
        gloss = None
        if self.use_gloss:
            gloss = self.gloss_branch.generate(memory, memory_mask, bos_id, eos_id, max_length, beam_size)
            memory = torch.cat([memory, self.gloss_branch.embedding(gloss)], 1)
            memory_mask = torch.cat([memory_mask, gloss.eq(self.pad_id)], 1)
        text = self.text_branch.generate(memory, memory_mask, bos_id, eos_id, max_length, beam_size)
        return gloss, text
