import torch
from torch import nn
from torch.nn import functional as F
from .prompt_injected_sign_transformer import PromptInjectedSignTransformer
from .hierarchical_prompt_bank import HierarchicalCrossLingualPromptBank
from .tap_module import TemporalAwarePromptInjection
from .prompt_routing_mechanism import PromptRoutingMechanism
from .prompt_forgetting_module import PromptForgettingModule
from .decoder import GlossTextDecoder

class UniSignPrompt(nn.Module):
    def __init__(self, visual_dim=1024, prompt_dim=768, hidden_dim=768,
                 num_layers=12, num_heads=12, num_signers=0, gloss_vocab_size=1066,
                 text_vocab_size=5000, num_languages=4, family_prompt_count=12,
                 signer_prompt_count=8, num_segments=4, top_k=5, tau=1.0,
                 beta=.1, gamma=.1, decoder_layers=2, use_gloss=False,
                 pad_id=0, bos_id=1, eos_id=2, max_seq_length=64, patch_size=16, use_tap=True, use_routing=True,
                 use_hclpb=True, use_pfm=True):
        super().__init__()
        if min(num_segments, family_prompt_count, signer_prompt_count) < 1:
            raise ValueError("Prompt counts and segment count must be positive")
        self.num_signers, self.num_languages = num_signers, num_languages
        self.use_tap, self.use_routing = use_tap, use_routing
        self.use_hclpb, self.use_pfm = use_hclpb, use_pfm
        self.bos_id, self.eos_id, self.max_seq_length = bos_id, eos_id, max_seq_length
        self.pi_st = PromptInjectedSignTransformer(visual_dim, prompt_dim, hidden_dim, num_layers, num_heads, patch_size=patch_size)
        self.clpb = HierarchicalCrossLingualPromptBank(range(num_languages), range(num_signers), prompt_dim, family_prompt_count, signer_prompt_count, gamma)
        self.tap = TemporalAwarePromptInjection(num_segments, prompt_dim, prompt_dim, prompt_dim, prompt_dim, family_prompt_count)
        self.prm = PromptRoutingMechanism(prompt_dim, 2*prompt_dim, family_prompt_count+signer_prompt_count, tau, top_k)
        self.pfm = PromptForgettingModule(hidden_dim, num_signers, beta) if num_signers else None
        self.decoder = GlossTextDecoder(hidden_dim, gloss_vocab_size, text_vocab_size, num_heads, decoder_layers, pad_id, use_gloss)

    def encode(self, visual, language_ids, signer_ids=None, frame_mask=None):
        if language_ids is None:
            raise ValueError("Language IDs are required")
        # eval() unconditionally discards signer metadata, including supplied labels.
        signer_ids = signer_ids if self.training and self.num_signers else None
        embedded, frame_mask = self.pi_st.embed(visual, frame_mask)
        valid = (~frame_mask).to(embedded.dtype)
        summary = (embedded.mean(2) * valid[:, :, None]).sum(1) / valid.sum(1, keepdim=True)
        bank, language, signer = self.clpb(language_ids, signer_ids)
        if not self.use_hclpb:
            language = self.clpb.language_bank[None].expand(visual.size(0), -1, -1)
            signer = None
            bank = language
        language_embedding = self.clpb.language_embed(language_ids)
        active, route_loss, scores, indices = self.prm(torch.cat([summary, language_embedding], -1), bank)
        if not self.use_routing:
            active = bank
            route_loss = bank.new_zeros(())
            indices = torch.arange(bank.size(1), device=bank.device)[None].expand(bank.size(0), -1)
            scores = bank.new_full(bank.shape[:2], 1 / bank.size(1))
        temporal = self.tap(language_embedding) if self.use_tap else []
        combined = [torch.cat([active, segment], 1) for segment in temporal] if self.use_tap else [active]
        features, mask = self.pi_st(embedded, combined, frame_mask)
        pooled = features.sum(1) / (~mask).sum(1, keepdim=True)
        # Sum over distinct languages and segments, as in Eqs. 23 and 28.
        unique_language = language_ids.unique()
        unique_embedding = self.clpb.language_embed(unique_language)
        _, regularized_language, _ = self.clpb(unique_language)
        regularized_temporal = self.tap(unique_embedding)
        align = sum((F.normalize(regularized_language, dim=-1)-F.normalize(p, dim=-1)).square().sum() for p in regularized_temporal)
        temp = sum((F.normalize(a, dim=-1)-F.normalize(b, dim=-1)).square().sum() for a,b in zip(regularized_temporal, regularized_temporal[1:]))
        zero = pooled.new_zeros(())
        if not self.use_tap:
            align, temp = zero, zero
        if not self.use_hclpb:
            align = zero
        forgetting = self.pfm(pooled, language, signer, signer_ids) if signer_ids is not None and self.use_pfm else None
        return {'features': features, 'pooled_features': pooled, 'memory_mask': mask,
                'routing_scores': scores, 'routing_indices': indices,
                'forgetting_loss': forgetting['total_forgetting_loss'] if forgetting else zero,
                'signer_logits': forgetting['signer_logits'] if forgetting else None,
                'alignment_loss': align, 'temporal_loss': temp if torch.is_tensor(temp) else zero,
                'routing_loss': route_loss, 'prompt_loss': self.clpb.prompt_regularization_loss()}

    def forward(self, visual_features, signer_onehot=None, language_onehot=None,
                *, language_ids=None, signer_ids=None, text_inputs=None,
                gloss_inputs=None, frame_mask=None):
        if language_ids is None and language_onehot is not None:
            language_ids = language_onehot.argmax(-1)
        if signer_ids is None and signer_onehot is not None:
            signer_ids = signer_onehot.argmax(-1)
        output = self.encode(visual_features, language_ids, signer_ids, frame_mask)
        gloss, text = self.decoder(output['features'], gloss_inputs, text_inputs, output['memory_mask'])
        output.update(gloss_logits=gloss, text_logits=text)
        return output

    @torch.no_grad()
    def generate(self, visual, language_ids, frame_mask=None, beam_size=5, max_length=None):
        if self.training:
            raise RuntimeError("Call model.eval() before deployment generation")
        if beam_size < 1:
            raise ValueError("beam_size must be positive")
        output = self.encode(visual, language_ids, frame_mask=frame_mask)
        gloss, text = self.decoder.generate(output['features'], output['memory_mask'], self.bos_id,
            self.eos_id, max_length or self.max_seq_length, beam_size)
        output.update(gloss_tokens=gloss, text_tokens=text)
        return output

    def load_backbone(self, path):
        state = torch.load(path, map_location='cpu', weights_only=True)
        expected = {k: v for k,v in self.pi_st.state_dict().items()
                    if k.startswith(('patch_embed.', 'input_proj.', 'layers.'))}
        if set(state) != set(expected):
            raise ValueError("Backbone checkpoint must exactly contain pi_st patch_embed, input_proj, and layers keys; see docs/MANUSCRIPT_ALIGNMENT.md")
        self.pi_st.load_state_dict(state, strict=False)
