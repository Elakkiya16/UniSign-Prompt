import argparse
import json
import statistics

def main():
    parser=argparse.ArgumentParser(description='Mean and sample SD across independently trained seeds')
    parser.add_argument('results',nargs='+')
    args=parser.parse_args()
    runs=[]
    for path in args.results:
        with open(path) as stream: runs.append(json.load(stream))
    seeds=[r.get('seed') for r in runs]
    if len(seeds)<2 or None in seeds or len(set(seeds))!=len(seeds):
        parser.error('Supply at least two results with distinct recorded seeds')
    common=set.intersection(*(set(r['metrics']) for r in runs))
    print(json.dumps({k:{'mean':statistics.mean(r['metrics'][k] for r in runs),
        'sample_sd':statistics.stdev(r['metrics'][k] for r in runs)} for k in sorted(common)},indent=2))

if __name__=='__main__': main()
