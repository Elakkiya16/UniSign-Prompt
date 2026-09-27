"""Metrics on decoded text/gloss, never on vocabulary ID strings."""
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from jiwer import wer

def rouge_l(reference, hypothesis):
    previous = [0] * (len(hypothesis)+1)
    for r in reference:
        current = [0]
        for j,h in enumerate(hypothesis):
            current.append(previous[j]+1 if r == h else max(previous[j+1],current[-1]))
        previous = current
    lcs = previous[-1]
    return 2*lcs/(len(reference)+len(hypothesis)) if reference or hypothesis else 1.0

def translation_metrics(references, hypotheses, gloss_references=None, gloss_hypotheses=None, chinese=False, include_meteor=True):
    if not references or len(references) != len(hypotheses):
        raise ValueError("Expected equal nonempty reference/hypothesis lists")
    tokenize = (lambda s: list(''.join(s.split()))) if chinese else str.split
    refs, hyps = list(map(tokenize, references)), list(map(tokenize,hypotheses))
    result = {f'BLEU-{n}':100*corpus_bleu([[r] for r in refs],hyps,weights=(1/n,)*n) for n in range(1,5)}
    result['ROUGE-L'] = 100*sum(rouge_l(r,h) for r,h in zip(refs,hyps))/len(refs)
    if include_meteor:
        # WordNet must be explicitly installed by the operator; never downloaded at evaluation time.
        result['METEOR'] = 100*sum(meteor_score([r],h) for r,h in zip(refs,hyps))/len(refs)
    if gloss_references is not None:
        result['Gloss-WER'] = 100*wer(gloss_references,gloss_hypotheses)
    else:
        result['Text-WER'] = 100*wer([' '.join(r) for r in refs],[' '.join(h) for h in hyps])
    return result

class TokenDecoder:
    def __init__(self, tokenizer_path, gloss_vocab_path=None, special_ids=(0,1,2)):
        from tokenizers import Tokenizer
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.special_ids = set(special_ids)
        self.gloss_vocab = None
        if gloss_vocab_path:
            with open(gloss_vocab_path,encoding='utf-8') as stream:
                self.gloss_vocab = [line.rstrip('\n') for line in stream]
    def clean(self,tokens):
        result = []
        for token in tokens:
            if token == 2:
                break
            if token not in self.special_ids:
                result.append(token)
        return result
    def text(self,tokens):
        return self.tokenizer.decode(self.clean(tokens), skip_special_tokens=True)
    def gloss(self,tokens):
        if self.gloss_vocab is None:
            raise ValueError("Configure a gloss vocabulary to compute Gloss-WER")
        return ' '.join(self.gloss_vocab[i] for i in self.clean(tokens))
