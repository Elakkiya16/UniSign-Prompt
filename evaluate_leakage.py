"""Independent post-hoc signer probe on deployment representations (Sec. III/V)."""
import argparse
import json
from pathlib import Path
import torch
from torch import nn
from runtime import restore, seed_everything, device_for, make_loader, move_batch

class SignerProbe(nn.Sequential):
    def __init__(self, dimension, num_signers):
        super().__init__(nn.Linear(dimension,256),nn.ReLU(),nn.Dropout(.3),nn.Linear(256,num_signers))

@torch.no_grad()
def extract(model,loader,device):
    model.eval()
    features,labels,ids = [],[],[]
    for raw in loader:
        batch = move_batch(raw,device)
        if batch['signer_ids'] is None:
            raise ValueError("Signer leakage requires real signer annotations")
        result = model.encode(batch['visual'],batch['language_ids'],frame_mask=batch['frame_mask'])
        features.append(result['pooled_features'].cpu())
        labels.append(batch['signer_ids'].cpu())
        ids.extend(raw['sample_id'])
    return torch.cat(features),torch.cat(labels),ids

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--train-split',default='train')
    parser.add_argument('--test-split',default='test')
    parser.add_argument('--device',default='auto')
    parser.add_argument('--epochs',type=int,default=50,help='Implementation choice; fix before evaluating test data')
    parser.add_argument('--learning-rate',type=float,default=.001)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--output',default='results/leakage.json')
    args = parser.parse_args()
    seed_everything(args.seed)
    device = device_for(args.device)
    model,checkpoint = restore(args.checkpoint,device)
    config = checkpoint['config']
    if not model.num_signers:
        raise ValueError('Signer leakage is not reported for datasets without suitable signer labels')
    x,y,train_ids = extract(model,make_loader(config,args.train_split,require_signers=True),device)
    tx,ty,test_ids = extract(model,make_loader(config,args.test_split,require_signers=True),device)
    if set(train_ids) & set(test_ids):
        raise ValueError('Probe train/test sample IDs overlap')
    if not set(ty.tolist()) <= set(y.tolist()):
        raise ValueError('Closed-set signer probe requires test signer classes represented in its training split')
    probe = SignerProbe(x.size(1),model.num_signers).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(),lr=args.learning_rate)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x,y),batch_size=128,shuffle=True)
    for _ in range(args.epochs):
        probe.train()
        for features,labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(probe(features.to(device)),labels.to(device))
            loss.backward()
            optimizer.step()
    probe.eval()
    predictions=[]
    with torch.no_grad():
        for features in tx.split(128):
            predictions.append(probe(features.to(device)).argmax(-1).cpu())
    accuracy=100*(torch.cat(predictions)==ty).float().mean().item()
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps({'signer_accuracy':accuracy,'chance_accuracy':100/model.num_signers,
        'train_samples':len(y),'test_samples':len(ty),'protocol':vars(args)},indent=2))
    torch.save(probe.state_dict(),output.with_suffix('.probe.pt'))
    print(f'Post-hoc signer accuracy: {accuracy:.2f}%')

if __name__ == '__main__':
    main()
