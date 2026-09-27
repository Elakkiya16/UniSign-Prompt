"""Fit BPE on training text only; share one English tokenizer across ASL/ISL."""
import argparse
from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--corpora',nargs='+',required=True,help='UTF-8 training transcripts, one sentence per line')
    parser.add_argument('--output',required=True)
    parser.add_argument('--vocab-size',type=int,default=5000)
    args=parser.parse_args()
    tokenizer=Tokenizer(models.BPE(unk_token='<unk>'))
    tokenizer.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder=decoders.ByteLevel()
    tokenizer.train(args.corpora,trainers.BpeTrainer(vocab_size=args.vocab_size,
        special_tokens=['<pad>','<bos>','<eos>','<unk>'],initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    tokenizer.save(args.output)
    print(f'Wrote {tokenizer.get_vocab_size()} tokens; reserve IDs 0/1/2 for PAD/BOS/EOS')

if __name__=='__main__': main()
