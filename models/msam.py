from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

class MSAM(nn.Module):
    def __init__(self, in_dim: int, dsm_dim: int=64, levels: int=4):
        super().__init__()
        self.dsm_dim=dsm_dim
        self.dsm_conv=nn.Sequential(nn.Conv2d(1,dsm_dim,3,padding=1),nn.GELU(),nn.Conv2d(dsm_dim,dsm_dim,3,padding=1))
        self.align=nn.ModuleList([nn.Conv2d(dsm_dim,in_dim,1) for _ in range(levels)])

    def forward(self, visual: list[torch.Tensor], dsm: torch.Tensor|None):
        b,_,h,w=visual[0].shape
        if dsm is None:
            zeros=visual[0].new_zeros((b,self.dsm_dim,h,w))
            return list(visual),zeros,False
        if dsm.shape[:2] != (b,1):
            raise ValueError('DSM expected [B,1,H,W].')
        dsm=F.interpolate(dsm,size=(h,w),mode='bilinear',align_corners=False)
        f_dsm=self.dsm_conv(dsm)
        fused=[]
        for feat,proj in zip(visual,self.align):
            aligned=F.interpolate(f_dsm,size=feat.shape[-2:],mode='bilinear',align_corners=False)
            fused.append(feat+proj(aligned))
        return fused,f_dsm,True
