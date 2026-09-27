import argparse
import csv
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer

def main():
    parser=argparse.ArgumentParser(description='Create BOS/text/EOS arrays from sample_id,text CSV')
    parser.add_argument('--csv',required=True)
    parser.add_argument('--tokenizer',required=True)
    parser.add_argument('--output-dir',required=True)
    args=parser.parse_args()
    tokenizer=Tokenizer.from_file(args.tokenizer)
    destination=Path(args.output_dir).resolve()
    destination.mkdir(parents=True,exist_ok=True)
    with open(args.csv,newline='',encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            path=(destination/(row['sample_id']+'.npy')).resolve()
            if path.parent!=destination:
                raise ValueError('sample_id must be a filename without directory components')
            np.save(path,np.array([1]+tokenizer.encode(row['text']).ids+[2],dtype=np.int64))

if __name__=='__main__': main()
