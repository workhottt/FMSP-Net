from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F

TAP_LAYERS = (6, 12, 18, 24)

class FrozenSAMViTH(nn.Module):
    """Direct access to the ORIGINAL SAM image encoder blocks, not a GRU proxy."""
    def __init__(self, checkpoint: str, input_size: int = 1024):
        super().__init__()
        if not checkpoint:
            raise ValueError('SAM ViT-H requires --sam-checkpoint (official sam_vit_h_*.pth).')
        try:
            from segment_anything import sam_model_registry
        except ImportError as exc:
            raise ImportError('Install official segment_anything from Meta first.') from exc
        sam = sam_model_registry['vit_h'](checkpoint=checkpoint)
        self.image_encoder = sam.image_encoder
        self.embedding_dim = 1280
        self.tap_layers = TAP_LAYERS
        self.input_size = input_size
        for p in self.image_encoder.parameters():
            p.requires_grad_(False)
        self.image_encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.image_encoder.eval()
        return self

    def forward(self, image: torch.Tensor) -> list[torch.Tensor]:
        # image: float [0,1], 3 channels, already spatially prepared/aligned.
        if image.shape[1] != 3:
            raise ValueError('SAM expects a 3-channel optical tensor.')
        enc = self.image_encoder
        # SAM expects input standardized with pixel_mean and pixel_std in 0..255 units.
        pixel_mean = torch.tensor([123.675,116.28,103.53],device=image.device,dtype=image.dtype)[None,:,None,None]
        pixel_std = torch.tensor([58.395,57.12,57.375],device=image.device,dtype=image.dtype)[None,:,None,None]
        image = (image * 255. - pixel_mean) / pixel_std
        if image.shape[-2:] != (self.input_size,self.input_size):
            # Resizing input is ONLY for ViT token sizing, NOT an original-data preprocessing script.
            image = F.interpolate(image, size=(self.input_size,self.input_size), mode='bilinear', align_corners=False)
        with torch.no_grad():
            x = enc.patch_embed(image)  # B,H',W',C
            if enc.pos_embed is not None:
                pos = enc.pos_embed
                if pos.shape[1:3] != x.shape[1:3]:
                    pos = F.interpolate(pos.permute(0,3,1,2),size=x.shape[1:3],mode='bicubic',align_corners=False).permute(0,2,3,1)
                x = x + pos
            taps = []
            for idx, block in enumerate(enc.blocks, start=1):
                x = block(x)
                if idx in self.tap_layers:
                    taps.append(x.permute(0,3,1,2).contiguous())
        if len(taps) != 4:
            raise RuntimeError('Could not extract all four SAM intermediate layers.')
        return taps

class ReferenceViT(nn.Module):
    """24-layer transformer SHAPED like SAM taps, randomly initialized.
    Test backend only. Must NEVER be called 'pretrained SAM' in reporting.
    """
    def __init__(self, width: int = 32, patch_size: int = 16, depth: int = 24):
        super().__init__()
        if depth < 24:
            raise ValueError('ReferenceViT needs at least 24 blocks for 6/12/18/24 taps.')
        self.patch_size = patch_size
        self.embedding_dim = width
        self.tap_layers = TAP_LAYERS
        self.patch_embed = nn.Conv2d(3, width, patch_size, stride=patch_size)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=width,nhead=4,dim_feedforward=width*2,
                                       dropout=0.,activation='gelu',batch_first=True,norm_first=True)
            for _ in range(depth)
        ])
        self.final_norm = nn.LayerNorm(width)
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        super().train(False)
        return self

    def forward(self, image: torch.Tensor) -> list[torch.Tensor]:
        with torch.no_grad():
            z = self.patch_embed(image)
            b,c,h,w = z.shape
            x = z.flatten(2).transpose(1,2)
            # Explicit fixed sinusoidal position encoding, deterministic.
            positions = torch.arange(h*w,device=x.device,dtype=x.dtype)[:,None]
            div = torch.exp(torch.arange(0,c,2,device=x.device,dtype=x.dtype)*(-math.log(10000.0)/c))
            pe = torch.zeros(1,h*w,c,device=x.device,dtype=x.dtype)
            pe[...,0::2] = torch.sin(positions*div)
            pe[...,1::2] = torch.cos(positions*div)
            x = x + pe
            outs = []
            for idx, block in enumerate(self.blocks,start=1):
                x = block(x)
                if idx in self.tap_layers:
                    outs.append(self.final_norm(x).transpose(1,2).reshape(b,c,h,w))
        return outs
