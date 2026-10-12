from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F

class CrossPromptDecoder(nn.Module):
    def __init__(self,dim:int,num_classes:int,dsm_dim:int=64):
        super().__init__()
        self.feature_merge=nn.Sequential(nn.Conv2d(dim*2,dim,3,padding=1),nn.GELU(),nn.Conv2d(dim,dim,3,padding=1))
        self.dsm_proj=nn.Conv2d(dsm_dim,dim,1,bias=False)
        self.q_proj=nn.Linear(dim,dim)
        self.key=nn.Conv2d(dim,dim,1)
        self.value=nn.Conv2d(dim,dim,1)
        self.cross=nn.MultiheadAttention(dim,num_heads=4,batch_first=True)
        self.pixel_proj=nn.Conv2d(dim,dim,1)
        self.seg_head=nn.Conv2d(dim,num_classes,1)

    def forward(self,refined,semantic_prompt,text_prototypes,dsm_features,dsm_active,output_size):
        fused=self.feature_merge(torch.cat([refined,semantic_prompt],dim=1))
        if dsm_active:
            h,w=fused.shape[-2:]
            fused=fused+self.dsm_proj(F.interpolate(dsm_features,size=(h,w),mode='bilinear',align_corners=False))
        b,c,h,w=fused.shape
        # K text semantic queries cross-attend to HW visual tokens.
        text_q=self.q_proj(text_prototypes).unsqueeze(0).expand(b,-1,-1)
        keys=self.key(fused).flatten(2).transpose(1,2)
        values=self.value(fused).flatten(2).transpose(1,2)
        class_ctx,_=self.cross(text_q,keys,values,need_weights=False)
        class_ctx=class_ctx+text_q
        pixels=self.pixel_proj(fused).flatten(2).transpose(1,2)
        class_logits=torch.einsum('bnc,bkc->bkn',pixels,class_ctx).reshape(b,-1,h,w)/math.sqrt(c)
        logits=self.seg_head(fused)+class_logits
        return F.interpolate(logits,size=output_size,mode='bilinear',align_corners=False)
