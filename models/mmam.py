from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

class SelectiveSSMReference(nn.Module):
    """Explicit diagonal input-conditioned state-space discretization.
    Computationally O(L) but Python scanning is slow; CPU smoke testing only.
    """
    def __init__(self,dim:int,d_state:int=16):
        super().__init__()
        self.dim,self.d_state=dim,d_state
        self.in_proj=nn.Linear(dim,dim*2)
        self.dt_proj=nn.Linear(dim,dim)
        self.b_proj=nn.Linear(dim,d_state)
        self.c_proj=nn.Linear(dim,d_state)
        self.log_a=nn.Parameter(torch.log(torch.arange(1,d_state+1,dtype=torch.float32)).repeat(dim,1))
        self.d=nn.Parameter(torch.ones(dim))
        self.out_proj=nn.Linear(dim,dim)

    def forward(self,x:torch.Tensor)->torch.Tensor:
        b,length,dim=x.shape
        u,z=self.in_proj(x).chunk(2,dim=-1)
        u=F.silu(u)
        dt=F.softplus(self.dt_proj(x))
        bt=self.b_proj(x)
        ct=self.c_proj(x)
        a=-torch.exp(self.log_a) # dim,state
        h=x.new_zeros((b,dim,self.d_state))
        outputs=[]
        for t in range(length):
            delta=dt[:,t,:,None]
            decay=torch.exp(delta*a[None,:,:])
            h=decay*h+delta*bt[:,t,None,:]*u[:,t,:,None]
            yt=(h*ct[:,t,None,:]).sum(-1)+self.d*u[:,t,:]
            outputs.append(yt*F.silu(z[:,t,:]))
        return self.out_proj(torch.stack(outputs,dim=1))

class BiDirectionalSSM(nn.Module):
    def __init__(self,dim:int=256,d_state:int=16,backend:str='mamba'):
        super().__init__()
        self.norm=nn.LayerNorm(dim)
        if backend=='mamba':
            try:
                from mamba_ssm import Mamba
            except ImportError as exc:
                raise ImportError('Production MMAM requires mamba_ssm and compatible CUDA wheels.') from exc
            self.forward_scan=Mamba(d_model=dim,d_state=d_state,d_conv=4,expand=2)
            self.backward_scan=Mamba(d_model=dim,d_state=d_state,d_conv=4,expand=2)
        elif backend=='reference':
            self.forward_scan=SelectiveSSMReference(dim,d_state)
            self.backward_scan=SelectiveSSMReference(dim,d_state)
        else:
            raise ValueError('backend must be mamba or reference')
        self.fuse=nn.Linear(2*dim,dim)

    def forward(self,x:torch.Tensor)->torch.Tensor:
        z=self.norm(x)
        fw=self.forward_scan(z)
        bw=self.backward_scan(z.flip(dims=(1,))).flip(dims=(1,))
        return x+self.fuse(torch.cat([fw,bw],dim=-1))

class MMAM(nn.Module):
    def __init__(self,in_dim:int,dim:int=256,d_state:int=16,backend:str='mamba',n_blocks:int=1):
        super().__init__()
        self.channels=dim
        self.project_stages=nn.ModuleList([nn.Conv2d(in_dim,dim,1) for _ in range(4)])
        self.blocks=nn.ModuleList([BiDirectionalSSM(dim,d_state,backend) for _ in range(n_blocks)])
        self.alignment=nn.Conv2d(dim,dim,1)

    def forward(self,stages:list[torch.Tensor])->torch.Tensor:
        if len(stages)!=4:
            raise ValueError('MMAM expects 4 SAM features: F6, F12, F18, F24.')
        h,w=stages[0].shape[-2:]
        tokens=[]
        for f,project in zip(stages,self.project_stages):
            f=project(F.interpolate(f,size=(h,w),mode='bilinear',align_corners=False))
            tokens.append(f.flatten(2).transpose(1,2))
        z=torch.cat(tokens,dim=1) # [B,4HW,Cv]
        for block in self.blocks:
            z=block(z)
        # Paper does not specify reducing 4HW to HW in Eq.(18).
        # We fuse the 4 processed stage outputs by their arithmetic mean.
        b=z.size(0)
        merged=z.reshape(b,4,h*w,self.channels).mean(dim=1)
        merged=merged.transpose(1,2).reshape(b,self.channels,h,w)
        return self.alignment(merged)
