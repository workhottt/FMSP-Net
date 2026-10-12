import math
import torch
from torch import nn
from torch.nn import functional as F

class LSAM(nn.Module):
    def __init__(self,dim:int=256,text_dim:int=512):
        super().__init__()
        self.text_projection=nn.Linear(text_dim,dim,bias=False)
        self.query=nn.Conv2d(dim,dim,1,bias=False)
        self.key=nn.Linear(dim,dim)
        self.value=nn.Linear(dim,dim)
        self.prompt_proj=nn.Sequential(nn.Conv2d(dim,dim,3,padding=1),nn.GELU(),nn.Conv2d(dim,dim,1))
        self.log_temperature=nn.Parameter(torch.tensor(0.0))

    def forward(self,features:torch.Tensor,text_embeddings:torch.Tensor):
        b,c,h,w=features.shape
        class_proto=self.text_projection(text_embeddings) # [K,C]
        q=self.query(features).flatten(2).transpose(1,2)
        kt=self.key(class_proto)
        vt=self.value(class_proto)
        temp=F.softplus(self.log_temperature)+1e-4
        attn=torch.softmax((q@kt.t())/(math.sqrt(c)*temp),dim=-1) # B,HW,K
        prompt=(attn@vt).transpose(1,2).reshape(b,c,h,w)
        prompt=self.prompt_proj(prompt)
        return prompt,attn.transpose(1,2).reshape(b,-1,h,w),class_proto
