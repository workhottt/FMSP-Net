from __future__ import annotations
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from checkpointing import load_weights, model_weights, read_checkpoint
from configs.datasets import get_dataset
from data import TensorSegDataset, SyntheticSmokeDataset
from metrics import confusion_update, scores
from models import FMSPNet, JointLoss


def parse_args():
    p = argparse.ArgumentParser(description='Train FMSP-Net from prepared .pt patches.')
    p.add_argument('--dataset', choices=['potsdam', 'vaihingen', 'loveda', 'uavid'], default='potsdam')
    p.add_argument('--train-manifest', help='JSON list of prepared training .pt samples')
    p.add_argument('--val-manifest', help='JSON list of prepared validation .pt samples')
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--batch-size', type=int, default=1, help='Physical batch size: start with 1 on A100 40GB')
    p.add_argument('--grad-accum-steps', type=int, default=1)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--amp', action='store_true', help='CUDA autocast fp16 with GradScaler')
    p.add_argument('--sam-checkpoint', default='', help='Official Meta SAM ViT-H .pth')
    p.add_argument('--sam-input-size', type=int, default=1024)
    p.add_argument('--clip-model', default='ViT-B-32')
    p.add_argument('--clip-pretrained', default='openai')
    p.add_argument('--backbone', choices=['sam', 'reference'], default='sam')
    p.add_argument('--ssm', choices=['mamba', 'reference'], default='mamba')
    p.add_argument('--text', choices=['open_clip', 'reference'], default='open_clip')
    p.add_argument('--feature-dim', type=int, default=256)
    p.add_argument('--dsm-dim', type=int, default=64)
    p.add_argument('--d-state', type=int, default=16)
    p.add_argument('--mamba-blocks', type=int, default=1)
    p.add_argument('--save-dir', default='checkpoints/a100_run')
    p.add_argument('--resume', default='')
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--smoke-test', action='store_true', help='CPU reference integration test only')
    p.add_argument('--smoke-size', type=int, default=32)
    return p.parse_args()


def build_model(cfg, args):
    if args.smoke_test:
        args.backbone = 'reference'
        args.ssm = 'reference'
        args.text = 'reference'
        args.feature_dim = min(args.feature_dim, 16)
        args.dsm_dim = min(args.dsm_dim, 8)
        args.d_state = min(args.d_state, 4)
    return FMSPNet(
        cfg, backbone_backend=args.backbone, mamba_backend=args.ssm,
        clip_backend=args.text, sam_checkpoint=args.sam_checkpoint,
        sam_input_size=args.sam_input_size, visual_dim=args.feature_dim,
        dsm_dim=args.dsm_dim, d_state=args.d_state,
        reference_width=16 if args.smoke_test else 32,
        clip_model=args.clip_model, clip_pretrained=args.clip_pretrained,
        mamba_blocks=args.mamba_blocks
    )


def run_epoch(model, loader, optimizer, criterion, device, cfg, training, use_amp=False, accum_steps=1):
    model.train(training)
    history = torch.zeros((len(cfg.classes), len(cfg.classes)), dtype=torch.int64, device=device)
    total_loss = 0.0
    n_batches = 0
    scaler = torch.cuda.amp.GradScaler(enabled=(training and use_amp))
    params = [p for p in model.parameters() if p.requires_grad]
    if training:
        optimizer.zero_grad(set_to_none=True)
    for i, batch in enumerate(loader):
        x = batch['image'].to(device, non_blocking=True)
        mask = batch['mask'].to(device, non_blocking=True)
        dsm = batch['dsm'].to(device, non_blocking=True) if cfg.has_dsm else None
        with torch.set_grad_enabled(training):
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(x, dsm, update_memory=training)
                loss, details = criterion(out, mask, model)
            if training:
                scaler.scale(loss / accum_steps).backward()
                if ((i + 1) % accum_steps == 0) or (i + 1 == len(loader)):
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(params, max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
        total_loss += float(loss.detach().cpu())
        n_batches += 1
        history = confusion_update(history, out['logits'].detach(), mask, cfg.ignore_index)
    report = scores(history)
    report['loss'] = total_loss / max(1, n_batches)
    return report


def main():
    args = parse_args()
    if args.grad_accum_steps < 1:
        raise ValueError('--grad-accum-steps must be >= 1')
    if args.smoke_test:
        args.epochs = 1
        args.batch_size = 1
        args.grad_accum_steps = 1
        args.device = 'cpu'
        args.amp = False
        args.workers = 0
        torch.set_num_threads(min(torch.get_num_threads(), 2))
    else:
        if not args.train_manifest:
            raise ValueError('Provide --train-manifest (prepared .pt patches), or use --smoke-test.')
        if any((args.backbone == 'reference', args.ssm == 'reference', args.text == 'reference')):
            raise ValueError('Production training requires --backbone sam --ssm mamba --text open_clip.')
        if args.device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable. Check NVIDIA driver, CUDA-enabled PyTorch and device.')
        if not args.sam_checkpoint:
            raise ValueError('Production SAM ViT-H requires --sam-checkpoint.')
    if args.amp and not args.device.startswith('cuda'):
        raise ValueError('--amp requires --device cuda')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    cfg = get_dataset(args.dataset)
    if args.smoke_test:
        train_ds = SyntheticSmokeDataset(1, args.smoke_size, len(cfg.classes), cfg.has_dsm)
        val_ds = train_ds
    else:
        train_ds = TensorSegDataset(args.train_manifest, cfg.has_dsm, len(cfg.classes))
        val_ds = TensorSegDataset(args.val_manifest, cfg.has_dsm, len(cfg.classes)) if args.val_manifest else None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                              pin_memory=args.device.startswith('cuda'))
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=args.workers,
                            pin_memory=args.device.startswith('cuda')) if val_ds is not None else None
    model = build_model(cfg, args).to(args.device)
    model.freeze_foundation()
    criterion = JointLoss(ignore_index=cfg.ignore_index)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.8)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    start_epoch = 0
    best = -1.0
    if args.resume:
        old = read_checkpoint(args.resume)
        if old['dataset'] != args.dataset:
            raise ValueError('Resume checkpoint dataset mismatch')
        load_weights(model, old)
        optimizer.load_state_dict(old['optimizer'])
        if 'scheduler' in old:
            scheduler.load_state_dict(old['scheduler'])
        start_epoch = int(old['epoch']) + 1
        best = float(old.get('best_mIoU', -1.0))
    for epoch in range(start_epoch, args.epochs):
        train_report = run_epoch(model, train_loader, optimizer, criterion, args.device, cfg,
                                 training=True, use_amp=args.amp, accum_steps=args.grad_accum_steps)
        val_report = run_epoch(model, val_loader, optimizer, criterion, args.device, cfg,
                               training=False, use_amp=args.amp) if val_loader is not None else train_report
        scheduler.step()
        print('epoch {}/{} train={} val={}'.format(epoch + 1, args.epochs,
              json.dumps(train_report), json.dumps(val_report)), flush=True)
        weights, adapter_only = model_weights(model)
        previous_best = best
        best = max(best, float(val_report['mIoU']))
        state = dict(state_dict=weights, adapter_only=adapter_only,
                     optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                     epoch=epoch, dataset=args.dataset, model_settings=model.model_settings,
                     validation=val_report, best_mIoU=best, is_smoke_test=bool(args.smoke_test))
        torch.save(state, save_dir / 'last.pt')
        if float(val_report['mIoU']) > previous_best:
            torch.save(state, save_dir / 'best.pt')
    print(f'Checkpoint: {save_dir / "best.pt"}')


if __name__ == '__main__':
    main()
