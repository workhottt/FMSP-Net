from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

class FSA(nn.Module):
    def __init__(self, channels:int, mask_grid:int=64):
        super().__init__()
        # Each layer gets separate frequency-domain learnable LOGITS.
        freq=torch.fft.fftfreq(mask_grid)
        fy,fx=torch.meshgrid(freq,freq,indexing='ij')
        radial=(fx.square()+fy.square()).sqrt()
        weight=torch.clamp(0.05+0.90*torch.exp(-radial.square()/0.055),0.001,0.999)
        self.mask_logits=nn.Parameter(torch.logit(weight)[None,None])
        self.conv_gate=nn.Conv2d(channels,channels,3,padding=1,groups=channels)
        self.conv_sharp=nn.Conv2d(channels,channels,3,padding=1)
        self.conv_mix=nn.Conv2d(channels,channels,1)
        self.project=nn.Conv2d(channels,channels,1)

    def frequency_mask(self,h:int,w:int)->torch.Tensor:
        # Interpolate in CENTERED frequency coordinates (not raw FFT ordering).
        shifted=torch.fft.fftshift(self.mask_logits,dim=(-2,-1))
        shifted=F.interpolate(shifted,size=(h,w),mode='bilinear',align_corners=False)
        logits=torch.fft.ifftshift(shifted,dim=(-2,-1))
        mask=torch.sigmoid(logits)
        # Conjugate symmetry: M(k) == M(-k) so IFFT is real-valued.
        conjugate=torch.roll(torch.flip(mask,(-2,-1)),(1,1),(-2,-1))
        return 0.5*(mask+conjugate)

    def forward(self,x:torch.Tensor)->torch.Tensor:
        original_dtype=x.dtype
        h,w=x.shape[-2:]
        # FFT in FP32 for mixed precision stability.
        spectrum=torch.fft.fft2(x.float(),dim=(-2,-1))
        mask=self.frequency_mask(h,w)
        low=torch.fft.ifft2(spectrum*mask,dim=(-2,-1)).real.to(original_dtype)
        high=torch.fft.ifft2(spectrum*(1-mask),dim=(-2,-1)).real.to(original_dtype)
        high_refined=high*torch.sigmoid(self.conv_gate(low))
        low_refined=low+self.conv_sharp(high)
        gate=torch.sigmoid(self.conv_mix(high_refined*low_refined))
        return x+self.project(gate*high_refined+(1-gate)*low_refined)
