"""Create a retained-training manifest from explicitly supplied forget sample IDs."""
import argparse
import csv
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True)
    parser.add_argument('--forget-ids',required=True,help='One exact sample_id per line')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    if Path(args.input).resolve()==Path(args.output).resolve():
        parser.error('Output must differ from original manifest')
    forget=set(Path(args.forget_ids).read_text().splitlines())
    with open(args.input,newline='',encoding='utf-8') as stream:
        reader=csv.DictReader(stream)
        fields=reader.fieldnames
        rows=list(reader)
    ids={r['sample_id'] for r in rows}
    if not forget or not forget<=ids:
        raise ValueError('Forget IDs must be nonempty and entirely present in training data')
    retained=[r for r in rows if r['sample_id'] not in forget]
    if not retained:
        raise ValueError('Forget set removes every training sample')
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        writer.writerows(retained)
    print(f'Retained {len(retained)} of {len(rows)} samples; retrain from the original backbone, not the forgotten-model checkpoint')

if __name__=='__main__': main()
