import argparse
import json
from pathlib import Path
import time
import torch
from metrics import TokenDecoder, translation_metrics
from runtime import restore, make_loader, move_batch, device_for, load_config, validate_vocabularies

def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'mps':
        torch.mps.synchronize()

@torch.no_grad()
def evaluate(model, loader, config, device, beam_size=5, include_meteor=True):
    model.eval()
    validate_vocabularies(config)
    decoder = TokenDecoder(config['data']['text_tokenizer'], config['data'].get('gloss_vocab') if model.decoder.use_gloss else None)
    rows, elapsed, count, active = [], 0., 0, 0
    warmed = False
    for raw in loader:
        batch = move_batch(raw,device)
        if not warmed:
            model.generate(batch['visual'],batch['language_ids'],batch['frame_mask'],beam_size)
            warmed = True
        synchronize(device)
        start = time.perf_counter()
        output = model.generate(batch['visual'],batch['language_ids'],batch['frame_mask'],beam_size)
        synchronize(device)
        elapsed += time.perf_counter()-start
        count += len(raw['sample_id'])
        active += output['routing_indices'].numel()
        for i,sample_id in enumerate(raw['sample_id']):
            row = {'sample_id':sample_id, 'reference':decoder.text(raw['text'][i].tolist()),
                   'hypothesis':decoder.text(output['text_tokens'][i].tolist())}
            if model.decoder.use_gloss:
                row.update(gloss_reference=decoder.gloss(raw['gloss'][i].tolist()),
                           gloss_hypothesis=decoder.gloss(output['gloss_tokens'][i].tolist()))
            rows.append(row)
    scores = translation_metrics([r['reference'] for r in rows],[r['hypothesis'] for r in rows],
        [r['gloss_reference'] for r in rows] if model.decoder.use_gloss else None,
        [r['gloss_hypothesis'] for r in rows] if model.decoder.use_gloss else None,
        chinese=config['data']['language']=='CSL', include_meteor=include_meteor)
    scores.update(latency_ms_per_sample=1000*elapsed/count,
                  active_prompt_ratio=active/(count*model.prm.num_prompts),
                  total_parameters=sum(p.numel() for p in model.parameters()),
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    return scores, rows

def main():
    parser = argparse.ArgumentParser(description='Signer-free beam-search evaluation')
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--split',default='test')
    parser.add_argument('--data-config',help='Override data paths/language for transfer; model remains exactly the checkpoint architecture')
    parser.add_argument('--device',default='auto')
    parser.add_argument('--beam-size',type=int,default=5)
    parser.add_argument('--output',default='results/evaluation.json')
    parser.add_argument('--skip-meteor',action='store_true',help='Explicitly omit METEOR if WordNet is unavailable')
    args = parser.parse_args()
    device = device_for(args.device)
    model, checkpoint = restore(args.checkpoint,device)
    config = checkpoint['config']
    if args.data_config:
        override = load_config(args.data_config)
        config['data'] = override['data']
    scores, rows = evaluate(model,make_loader(config,args.split),config,device,args.beam_size,not args.skip_meteor)
    output = Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps({'metrics':scores,'predictions':rows,'split':args.split,
        'checkpoint':args.checkpoint,'beam_size':args.beam_size,'seed':checkpoint.get('seed')},indent=2,ensure_ascii=False))
    print(json.dumps(scores,indent=2))

if __name__ == '__main__':
    main()
