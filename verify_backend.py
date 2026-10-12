from __future__ import annotations
import argparse
import torch
from configs.datasets import get_dataset
from models import FMSPNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sam-checkpoint', required=True)
    parser.add_argument('--dataset', choices=['potsdam', 'vaihingen', 'loveda', 'uavid'], default='potsdam')
    parser.add_argument('--patch-size', type=int, default=512)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available: install matching cu118 PyTorch and check the GPU driver.')
    cfg = get_dataset(args.dataset)
    print(f'PyTorch={torch.__version__} CUDA-build={torch.version.cuda} device={args.device}', flush=True)
    model = FMSPNet(cfg, backbone_backend='sam', mamba_backend='mamba', clip_backend='open_clip',
                    sam_checkpoint=args.sam_checkpoint, sam_input_size=1024,
                    visual_dim=256, dsm_dim=64, d_state=16).to(args.device).eval()
    image = torch.rand(1, 3, args.patch_size, args.patch_size, device=args.device)
    dsm = torch.rand(1, 1, args.patch_size, args.patch_size, device=args.device) if cfg.has_dsm else None
    with torch.inference_mode():
        pred = model(image, dsm)
    expected = (1, len(cfg.classes), args.patch_size, args.patch_size)
    assert tuple(pred['logits'].shape) == expected, (pred['logits'].shape, expected)
    assert bool(torch.isfinite(pred['logits']).all()), 'Non-finite logits'
    print(f'REAL-BACKEND FORWARD PASS OK: logits={expected}, DSM-active={pred["dsm_active"]}', flush=True)
    if args.device.startswith('cuda'):
        print(f'GPU peak allocated: {torch.cuda.max_memory_allocated()/(1024**3):.2f} GiB')


if __name__ == '__main__':
    main()
