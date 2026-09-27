import argparse
import json
from pathlib import Path
import torch
from models import UniSignPrompt
from losses import MultiObjectiveForgettingLoss
from runtime import load_config, seed_everything, device_for, make_loader, move_batch, model_forward, validate_vocabularies
from evaluate_extended import evaluate

def main():
    parser = argparse.ArgumentParser(description='Manuscript-aligned prompt tuning with a frozen encoder')
    parser.add_argument('--config',required=True)
    parser.add_argument('--device',default='auto')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--train-split',default='train')
    parser.add_argument('--val-split',default='val')
    parser.add_argument('--init-checkpoint',help='Full model initialization for transfer; target config controls supervision')
    args = parser.parse_args()
    config = load_config(args.config)
    validate_vocabularies(config)
    seed_everything(args.seed)
    device = device_for(args.device)
    model = UniSignPrompt(**config['model']).to(device)
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint,map_location='cpu',weights_only=True)
        source = checkpoint['model']
        target = model.state_dict()
        # Supervision-specific heads and signer embeddings may differ across languages.
        excluded = ('pfm.', 'clpb.signer_embed.', 'decoder.gloss_branch.')
        transferable = {k:v for k,v in source.items() if not k.startswith(excluded)}
        for key,value in transferable.items():
            if key not in target or value.shape != target[key].shape:
                raise ValueError(f"Incompatible transfer parameter {key}; ASL/ISL require a shared tokenizer and architecture")
        model.load_state_dict(transferable,strict=False)
    elif config['training'].get('backbone_checkpoint'):
        model.load_backbone(config['training']['backbone_checkpoint'])
    else:
        raise ValueError('Set training.backbone_checkpoint or supply --init-checkpoint')
    train_loader = make_loader(config,args.train_split,training=True)
    validation = make_loader(config,args.val_split)
    criterion = MultiObjectiveForgettingLoss(**config['loss'],pad_id=config['model'].get('pad_id',0))
    training = config['training']
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
        lr=training['learning_rate'],weight_decay=training['weight_decay'])
    steps = training['epochs']*len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:max(0.,1-step/max(1,steps)))
    destination = Path(training['output_dir'])/f'seed_{args.seed}'
    destination.mkdir(parents=True,exist_ok=True)
    best, history = float('-inf'), []
    for epoch in range(training['epochs']):
        model.train()
        total = 0.
        for raw in train_loader:
            batch = move_batch(raw,device)
            optimizer.zero_grad(set_to_none=True)
            output = model_forward(model,batch)
            losses = criterion(output,batch['text'][:,1:],batch['gloss'][:,1:] if batch['gloss'] is not None else None)
            losses['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],training['grad_clip_norm'])
            optimizer.step()
            scheduler.step()
            total += float(losses['total_loss'].detach())
        scores,_ = evaluate(model,validation,config,device,beam_size=5,include_meteor=False)
        record = {'epoch':epoch+1,'train_loss':total/len(train_loader),'validation':scores}
        history.append(record)
        print(json.dumps(record),flush=True)
        state = {'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                 'config':config,'seed':args.seed,'epoch':epoch+1}
        torch.save(state,destination/'last.pt')
        if scores['BLEU-4'] > best:
            best = scores['BLEU-4']
            torch.save(state,destination/'best.pt')
        (destination/'history.json').write_text(json.dumps(history,indent=2))

if __name__ == '__main__':
    main()
