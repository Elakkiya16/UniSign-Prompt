import argparse
import json
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description='Absolute percentage-point gaps to independently measured retraining')
    parser.add_argument('--signer',required=True)
    parser.add_argument('--retrained-signer',required=True)
    parser.add_argument('--mia',required=True)
    parser.add_argument('--retrained-mia',required=True)
    args=parser.parse_args()
    read=lambda path:json.loads(Path(path).read_text())
    sa,rsa=read(args.signer),read(args.retrained_signer)
    mia,rmia=read(args.mia),read(args.retrained_mia)
    if mia['protocol_note']!=rmia['protocol_note'] or mia['attack']!=rmia['attack']:
        raise ValueError('MIA populations and attack protocol must match')
    print(json.dumps({'delta_sa':abs(sa['signer_accuracy']-rsa['signer_accuracy']),
        'delta_mia':abs(mia['mia_success_rate']-rmia['mia_success_rate'])},indent=2))

if __name__=='__main__': main()
