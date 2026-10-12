from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


def dice_loss(logits,target,ignore_index=255,eps=1e-6):
    k=logits.size(1)
    valid=(target!=ignore_index)
    safe=target.masked_fill(~valid,0)
    probs=logits.softmax(1)*valid.unsqueeze(1)
    y=F.one_hot(safe.long(),k).permute(0,3,1,2)*valid.unsqueeze(1)
    num=2*(probs*y).sum(dim=(0,2,3)) + eps
    den=probs.sum(dim=(0,2,3)) + y.sum(dim=(0,2,3)) + eps
    active=y.sum(dim=(0,2,3))>0
    return (1-num[active]/den[active]).mean() if active.any() else logits.sum()*0


def boundary_distance_loss(logits,target,ignore_index=255):
    """Eq.(30) with D_G as distance to nearest GT region boundary.
    CPU numpy distance transform; no separate data preprocessing scripts needed.
    """
    try:
        from scipy.ndimage import distance_transform_edt, binary_erosion
    except ImportError as exc:
        raise ImportError('scipy is needed for exact distance-transform boundary loss.') from exc
    probs=logits.softmax(1)
    b,k,h,w=probs.shape
    valid=(target!=ignore_index)
    safe=target.masked_fill(~valid,0)
    true=F.one_hot(safe.long(),k).permute(0,3,1,2).to(probs.dtype)
    targets=safe.detach().cpu().numpy()
    masks=valid.detach().cpu().numpy()
    distances=np.zeros((b,k,h,w),dtype=np.float32)
    for bi in range(b):
        for ci in range(k):
            region=(targets[bi]==ci)&masks[bi]
            if not region.any():
                continue
            edge=region^binary_erosion(region)
            distances[bi,ci]=distance_transform_edt(~edge).astype(np.float32)
    weights=torch.from_numpy(distances).to(logits.device)
    # Normalize to avoid dataset-specific pixel dimensions dominating the scale.
    weights=weights/(weights.amax(dim=(2,3),keepdim=True).clamp(min=1))
    err=(probs-true).abs()*weights*valid[:,None,:].to(weights.dtype)
    return err.sum()/(valid.sum().clamp(min=1)*k)


def prototype_contrastive_loss(features,labels,memory,initialized,ignore_index=255,tau=0.07):
    if not bool(initialized.any()):
        return features.sum()*0
    labels=F.interpolate(labels.unsqueeze(1).float(),size=features.shape[-2:],mode='nearest')[:,0].long()
    valid=(labels!=ignore_index)&(labels>=0)&(labels<memory.size(0))
    # Do not supervise classes whose prototypes have not been initialized yet.
    safe=labels.clamp(0,memory.size(0)-1)
    valid=valid&initialized[safe]
    if not valid.any():
        return features.sum()*0
    f=F.normalize(features.permute(0,2,3,1)[valid],dim=-1)
    m=F.normalize(memory.detach(),dim=-1)
    logits=f@m.t()/tau
    logits[:,~initialized]=-1e4
    return F.cross_entropy(logits,safe[valid])

class JointLoss(nn.Module):
    def __init__(self,ignore_index=255,weights=(1.0,0.5,0.2,0.1)):
        super().__init__()
        self.ignore_index=ignore_index
        self.weights=weights

    def forward(self,outputs:dict,labels:torch.Tensor,model:nn.Module):
        logits=outputs['logits']
        valid=labels!=self.ignore_index
        if not valid.any():
            zero=logits.sum()*0
            return zero,dict(ce=zero,dice=zero,boundary=zero,proto=zero)
        ce=F.cross_entropy(logits,labels.long(),ignore_index=self.ignore_index)
        dice=dice_loss(logits,labels,self.ignore_index)
        boundary=boundary_distance_loss(logits,labels,self.ignore_index)
        proto=prototype_contrastive_loss(outputs['features'],labels,model.dpr.memory,
                                          model.dpr.initialized,self.ignore_index)
        losses=dict(ce=ce,dice=dice,boundary=boundary,proto=proto)
        total=sum(weight*loss for weight,loss in zip(self.weights,losses.values()))
        return total,losses
