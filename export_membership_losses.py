"""Export per-sample teacher-forced text CE under signer-free deployment encoding."""
import argparse
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from runtime import restore, make_loader, move_batch, model_forward, device_for

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--split',required=True)
    parser.add_argument('--member',type=int,choices=[0,1],required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='auto')
    args=parser.parse_args()
    device=device_for(args.device)
    model,checkpoint=restore(args.checkpoint,device)
    model.eval()
    rows=[]
    with torch.no_grad():
        for raw in make_loader(checkpoint['config'],args.split):
            batch=move_batch(raw,device)
            output=model_forward(model,batch)
            labels=batch['text'][:,1:]
            loss=F.cross_entropy(output['text_logits'].transpose(1,2),labels,
                reduction='none',ignore_index=model.decoder.pad_id)
            losses=loss.sum(1)/labels.ne(model.decoder.pad_id).sum(1)
            rows.extend(dict(sample_id=sample,member=args.member,loss=float(value))
                        for sample,value in zip(raw['sample_id'],losses))
    destination=Path(args.output)
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(rows,indent=2))

if __name__=='__main__': main()
