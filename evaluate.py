from __future__ import annotations
import argparse
import json
import torch
from torch.utils.data import DataLoader
from data import TensorSegDataset
from inference import load_model
from metrics import confusion_update, scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--sam-checkpoint', default=None)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--allow-smoke-checkpoint', action='store_true')
    args = parser.parse_args()
    model, cfg, state = load_model(args.checkpoint, args.device, args.sam_checkpoint)
    if state.get('is_smoke_test', False) and not args.allow_smoke_checkpoint:
        raise ValueError('Synthetic checkpoint is for mechanical tests only.')
    loader = DataLoader(TensorSegDataset(args.manifest, cfg.has_dsm, len(cfg.classes)), batch_size=1)
    confusion = torch.zeros((len(cfg.classes), len(cfg.classes)), dtype=torch.int64, device=args.device)
    for batch in loader:
        with torch.inference_mode():
            pred = model(batch['image'].to(args.device),
                         batch['dsm'].to(args.device) if cfg.has_dsm else None)
        confusion = confusion_update(confusion, pred['logits'], batch['mask'].to(args.device), cfg.ignore_index)
    print(json.dumps({'dataset': cfg.name, 'classes': cfg.classes, **scores(confusion)}, indent=2))


if __name__ == '__main__':
    main()
