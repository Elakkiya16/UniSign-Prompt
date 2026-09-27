"""Paired BLEU bootstrap over exported sentence predictions; no model retraining."""
import argparse
import json
import random
import statistics
from nltk.translate.bleu_score import corpus_bleu

def paired_bootstrap(ours, baseline, resamples=1000, seed=42, chinese=False):
    if len(ours)!=len(baseline) or not ours:
        raise ValueError('Paired nonempty predictions required')
    by_id={r['sample_id']:r for r in baseline}
    if len(by_id)!=len(baseline) or len({r['sample_id'] for r in ours})!=len(ours):
        raise ValueError('Duplicate sample IDs')
    baseline=[by_id[r['sample_id']] for r in ours]
    if any(a['reference']!=b['reference'] for a,b in zip(ours,baseline)):
        raise ValueError('References differ')
    tokenize=(lambda s:list(''.join(s.split()))) if chinese else str.split
    refs=[[tokenize(r['reference'])] for r in ours]
    a=[tokenize(r['hypothesis']) for r in ours]
    b=[tokenize(r['hypothesis']) for r in baseline]
    rng=random.Random(seed)
    differences=[]
    for _ in range(resamples):
        indices=[rng.randrange(len(ours)) for _ in ours]
        reference=[refs[i] for i in indices]
        differences.append(100*(corpus_bleu(reference,[a[i] for i in indices])-corpus_bleu(reference,[b[i] for i in indices])))
    differences.sort()
    return {'mean_resampled_delta_bleu4':statistics.mean(differences),
            'percentile_95_interval':[differences[int(.025*resamples)],differences[min(resamples-1,int(.975*resamples))]],
            'one_sided_bootstrap_p':(1+sum(d<=0 for d in differences))/(resamples+1),
            'resamples':resamples,'seed':seed}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('ours')
    parser.add_argument('baseline')
    parser.add_argument('--resamples',type=int,default=1000)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--chinese',action='store_true')
    args=parser.parse_args()
    if args.resamples<1:
        parser.error('resamples must be positive')
    with open(args.ours) as stream: ours=json.load(stream)['predictions']
    with open(args.baseline) as stream: baseline=json.load(stream)['predictions']
    print(json.dumps(paired_bootstrap(ours,baseline,args.resamples,args.seed,args.chinese),indent=2))

if __name__=='__main__': main()
