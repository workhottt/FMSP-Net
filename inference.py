from __future__ import annotations
import argparse
from pathlib import Path
import torch

from checkpointing import load_weights, read_checkpoint
from configs.datasets import get_dataset
from models import FMSPNet


def load_model(checkpoint_path, device='cpu', sam_checkpoint_override=None):
    state = read_checkpoint(checkpoint_path)
    cfg = get_dataset(state['dataset'])
    kw = state['model_settings'].copy()
    if sam_checkpoint_override:
        kw['sam_checkpoint'] = sam_checkpoint_override
    if state.get('adapter_only') and not kw.get('sam_checkpoint'):
        raise ValueError('Adapter-only weights require the original official SAM via --sam-checkpoint.')
    model = FMSPNet(cfg, **kw)
    load_weights(model, state)
    model.to(device).eval()
    return model, cfg, state


def infer_tensor(model, prepared, device='cpu'):
    image = prepared['image'].float()
    if image.ndim == 3:
        image = image.unsqueeze(0)
    dsm = prepared.get('dsm')
    if dsm is not None:
        dsm = dsm.float()
        if dsm.ndim == 3:
            dsm = dsm.unsqueeze(0)
    with torch.inference_mode():
        result = model(image.to(device), dsm.to(device) if dsm is not None else None)
        probs = result['logits'].softmax(dim=1)
    return {'prediction': probs.argmax(dim=1).cpu(),
            'probabilities': probs.cpu(),
            'activation': result['activation'].cpu()}


def main():
    parser = argparse.ArgumentParser(description='Predict class mask from prepared .pt tensor')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', default='examples/prediction.pt')
    parser.add_argument('--sam-checkpoint', default=None)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--allow-smoke-checkpoint', action='store_true')
    args = parser.parse_args()
    state = read_checkpoint(args.checkpoint)
    if state.get('is_smoke_test', False) and not args.allow_smoke_checkpoint:
        raise ValueError('Synthetic smoke checkpoint: pass --allow-smoke-checkpoint for testing only.')
    model, cfg, state = load_model(args.checkpoint, args.device, args.sam_checkpoint)
    prepared = torch.load(args.input, map_location='cpu', weights_only=True)
    outputs = infer_tensor(model, prepared, args.device)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(outputs, output)
    print(f'Saved prediction {tuple(outputs["prediction"].shape)}, dataset={cfg.name}, to {output}')


if __name__ == '__main__':
    main()
