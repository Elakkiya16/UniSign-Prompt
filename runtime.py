from functools import partial
from pathlib import Path
import random
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from datasets import ContinuousSignDataset, collate_samples
from models import UniSignPrompt

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def load_config(path):
    with open(path, encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    if tuple(config['model'].get(k, v) for k,v in [('pad_id',0),('bos_id',1),('eos_id',2)]) != (0,1,2):
        raise ValueError('This data protocol reserves PAD=0, BOS=1 and EOS=2')
    if bool(config['loss']['lambda_gloss']) != config['model']['use_gloss']:
        raise ValueError("Gloss branch and loss must agree")
    if config['data']['name'] == 'How2Sign' and (config['model']['use_gloss'] or config['model']['num_signers'] or config['loss']['lambda_forget']):
        raise ValueError("How2Sign must be gloss-free with forgetting disabled")
    return config

def device_for(name):
    if name != 'auto':
        return torch.device(name)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def make_loader(config, split, training=False, require_signers=False):
    data, model = config['data'], config['model']
    meta = data.get(split+'_meta')
    if not meta:
        raise ValueError(f"Configure data.{split}_meta before using this split")
    dataset = ContinuousSignDataset(meta, data['root'], data['language'], model['use_gloss'],
        (training or require_signers) and model['num_signers'] > 0, model['num_signers'],
        data['fps'], data['resolution'], model.get('bos_id',1), model.get('eos_id',2), model.get('pad_id',0),
        model['text_vocab_size'], model['gloss_vocab_size'])
    return DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=training,
        num_workers=config['training'].get('num_workers',0),
        collate_fn=partial(collate_samples, pad_id=model.get('pad_id',0)))

def move_batch(batch, device):
    return {k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}

def model_forward(model, batch):
    return model(batch['visual'], language_ids=batch['language_ids'], signer_ids=batch['signer_ids'],
        frame_mask=batch['frame_mask'], text_inputs=batch['text'][:,:-1],
        gloss_inputs=batch['gloss'][:,:-1] if batch['gloss'] is not None else None)

def restore(path, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model = UniSignPrompt(**checkpoint['config']['model']).to(device)
    model.load_state_dict(checkpoint['model'])
    return model, checkpoint


def validate_vocabularies(config):
    from tokenizers import Tokenizer
    data, model = config['data'], config['model']
    tokenizer = Tokenizer.from_file(data['text_tokenizer'])
    if tokenizer.get_vocab_size() != model['text_vocab_size']:
        raise ValueError('model.text_vocab_size must exactly match the prepared tokenizer')
    if [tokenizer.token_to_id(t) for t in ['<pad>', '<bos>', '<eos>']] != [0, 1, 2]:
        raise ValueError('Tokenizer must reserve PAD=0, BOS=1, EOS=2')
    if model['use_gloss']:
        vocabulary = Path(data['gloss_vocab']).read_text(encoding='utf-8').splitlines()
        if len(vocabulary) != model['gloss_vocab_size'] or vocabulary[:3] != ['<pad>', '<bos>', '<eos>']:
            raise ValueError('Prepare gloss vocabulary with PAD/BOS/EOS and matching model.gloss_vocab_size')
