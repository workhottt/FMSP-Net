from __future__ import annotations
import torch
from torch import nn
from .backbone import FrozenSAMViTH,ReferenceViT
from .text import FrozenCLIPText
from .msam import MSAM
from .fsa import FSA
from .mmam import MMAM
from .lsam import LSAM
from .dpr import DPR
from .decoder import CrossPromptDecoder
from configs.datasets import DatasetConfig

class FMSPNet(nn.Module):
    def __init__(self, dataset:DatasetConfig,backbone_backend='sam',mamba_backend='mamba',
                 clip_backend='open_clip',sam_checkpoint='',sam_input_size=1024,
                 visual_dim=256,dsm_dim=64,d_state=16,reference_width=32,
                 clip_model='ViT-B-32',clip_pretrained='openai',mamba_blocks=1):
        super().__init__()
        self.dataset=dataset
        self.model_settings=dict(
            backbone_backend=backbone_backend,mamba_backend=mamba_backend,
            clip_backend=clip_backend,sam_checkpoint=sam_checkpoint,
            sam_input_size=sam_input_size,visual_dim=visual_dim,dsm_dim=dsm_dim,
            d_state=d_state,reference_width=reference_width,
            clip_model=clip_model,clip_pretrained=clip_pretrained,mamba_blocks=mamba_blocks)
        if backbone_backend=='sam':
            self.backbone=FrozenSAMViTH(sam_checkpoint,input_size=sam_input_size)
        elif backbone_backend=='reference':
            self.backbone=ReferenceViT(width=reference_width)
        else:
            raise ValueError('backbone_backend must be sam or reference')
        c=self.backbone.embedding_dim
        self.msam=MSAM(c,dsm_dim)
        self.fsas=nn.ModuleList(FSA(c) for _ in range(4))
        self.mmam=MMAM(c,visual_dim,d_state,mamba_backend,n_blocks=mamba_blocks)
        self.text=FrozenCLIPText(dataset.prompts,backend=clip_backend,
                                 model_name=clip_model,pretrained=clip_pretrained)
        self.lsam=LSAM(visual_dim,self.text.out_dim)
        self.dpr=DPR(visual_dim,len(dataset.classes),momentum=0.999)
        self.decoder=CrossPromptDecoder(visual_dim,len(dataset.classes),dsm_dim)

    def forward(self,image:torch.Tensor,dsm:torch.Tensor|None=None,update_memory:bool=False):
        if image.ndim!=4 or image.shape[1]!=3:
            raise ValueError('image must be [B,3,H,W], normalized into [0,1] externally.')
        original_size=image.shape[-2:]
        base=self.backbone(image)
        anchored,f_dsm,dsm_active=self.msam(base,dsm if self.dataset.has_dsm else None)
        adapted=[fsa(f) for fsa,f in zip(self.fsas,anchored)]
        features=self.mmam(adapted)
        prompt,activation,classes=self.lsam(features,self.text())
        refined=self.dpr(features,activation,update_memory=(update_memory and self.training))
        logits=self.decoder(refined,prompt,classes,f_dsm,dsm_active,original_size)
        return dict(logits=logits,features=features,activation=activation,
                    dsm_active=dsm_active)

    def freeze_foundation(self):
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        for parameter in self.text.parameters():
            parameter.requires_grad_(False)
