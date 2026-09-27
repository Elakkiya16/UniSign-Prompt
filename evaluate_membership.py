"""Explicit loss-threshold MIA baseline; the manuscript does not specify its attack."""
import argparse
import json
from pathlib import Path
import numpy as np

def validate_rows(rows):
    ids=[row['sample_id'] for row in rows]
    labels=np.array([row['member'] for row in rows])
    losses=np.array([row['loss'] for row in rows],dtype=float)
    if len(ids)!=len(set(ids)) or not np.isfinite(losses).all():
        raise ValueError('Duplicate IDs or invalid losses')
    if set(labels.tolist())!={0,1} or sum(labels==0)!=sum(labels==1):
        raise ValueError('Use balanced member/nonmember samples with binary labels')
    return ids,labels,losses

def loss_threshold_attack(calibration, evaluation):
    calibration_ids,labels,losses=validate_rows(calibration)
    evaluation_ids,test_labels,test_losses=validate_rows(evaluation)
    if set(calibration_ids)&set(evaluation_ids):
        raise ValueError('Attack calibration/evaluation sample IDs must be disjoint')
    unique=np.unique(losses)
    thresholds=np.concatenate([[np.nextafter(unique[0],-np.inf)],(unique[:-1]+unique[1:])/2,[unique[-1]]])
    accuracies=[np.mean((losses<=threshold)==labels) for threshold in thresholds]
    threshold=float(thresholds[int(np.argmax(accuracies))])
    return {'mia_success_rate':100*float(np.mean((test_losses<=threshold)==test_labels)),
            'threshold':threshold,'calibration_samples':len(labels),'evaluation_samples':len(test_labels),
            'attack':'loss-threshold; lower loss predicts member; balanced accuracy equals accuracy'}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration',required=True,help='JSON list: sample_id, member (0/1), loss')
    parser.add_argument('--evaluation',required=True,help='Disjoint balanced JSON list from the same fixed model')
    parser.add_argument('--protocol-note',required=True,help='Record member/nonmember populations, loss, and forget subset')
    parser.add_argument('--output',default='results/membership.json')
    args=parser.parse_args()
    calibration=json.loads(Path(args.calibration).read_text())
    evaluation=json.loads(Path(args.evaluation).read_text())
    result=loss_threshold_attack(calibration,evaluation)
    result['protocol_note']=args.protocol_note
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
