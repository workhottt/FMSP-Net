import torch
from torch import nn
from torch.nn import functional as F

class DPR(nn.Module):
    def __init__(self,dim:int,num_classes:int,momentum:float=0.999):
        super().__init__()
        self.momentum=momentum
        self.num_classes=num_classes
        self.register_buffer('memory',torch.zeros(num_classes,dim))
        self.register_buffer('initialized',torch.zeros(num_classes,dtype=torch.bool))
        self.refine=nn.Sequential(nn.Conv2d(dim,dim,1),nn.GELU(),nn.Conv2d(dim,dim,1))

    @torch.no_grad()
    def update(self,feat:torch.Tensor,attention:torch.Tensor):
        # Softmax LSAM A, update based on argmax class pseudo assignments.
        b,c,h,w=feat.shape
        f=feat.detach().permute(0,2,3,1).reshape(-1,c)
        labels=attention.detach().argmax(dim=1).reshape(-1)
        for k in range(self.num_classes):
            selected=(labels==k)
            if selected.any():
                prototype=f[selected].mean(dim=0)
                if not bool(self.initialized[k]):
                    self.memory[k].copy_(prototype)
                    self.initialized[k]=True
                else:
                    self.memory[k].mul_(self.momentum).add_(prototype,alpha=1-self.momentum)

    def forward(self,feat:torch.Tensor,attention:torch.Tensor,update_memory:bool=False):
        if update_memory:
            self.update(feat,attention)
        b,c,h,w=feat.shape
        flat=feat.flatten(2).transpose(1,2)
        sim=flat@F.normalize(self.memory,dim=-1).t()
        sim=sim.masked_fill(~self.initialized[None,None,:],-1e4)
        # If no prototypes initialized, assign zero retrieval deterministically.
        if not bool(self.initialized.any()):
            retrieval=torch.zeros_like(flat)
        else:
            weights=sim.softmax(-1)
            retrieval=weights@self.memory
        retrieval=retrieval.transpose(1,2).reshape(b,c,h,w)
        return feat+self.refine(retrieval)
