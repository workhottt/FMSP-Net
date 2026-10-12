
from __future__ import annotations
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset

class TensorSegDataset(Dataset):
    def __init__(self,manifest:str,has_dsm:bool,num_classes:int):
        self.manifest=Path(manifest).resolve()
        with self.manifest.open(encoding='utf8') as fh:
            paths=json.load(fh)
        if not isinstance(paths,list) or not paths:
            raise ValueError('Manifest must be a nonempty JSON list of .pt file paths.')
        self.paths=[(self.manifest.parent/p).resolve() for p in paths]
        self.has_dsm=has_dsm
        self.num_classes=num_classes

    def __len__(self):return len(self.paths)

    def __getitem__(self,index):
        payload=torch.load(self.paths[index],map_location='cpu',weights_only=True)
        image=payload['image'].float()
        mask=payload['mask'].long()
        if image.ndim!=3 or image.shape[0]!=3 or image.shape[-2:]!=mask.shape:
            raise ValueError('Expected image [3,H,W] and mask [H,W] on identical grid.')
        if mask.ne(255).any() and (mask[mask!=255].min()<0 or mask[mask!=255].max()>=self.num_classes):
            raise ValueError(f'Class labels must be 0..{self.num_classes-1} or 255.')
        if image.min()<0 or image.max()>1:
            raise ValueError('Images must already be normalized to [0,1].')
        if self.has_dsm:
            if 'dsm' not in payload:
                raise ValueError(f'DSM is required but not found: {self.paths[index]}')
            dsm=payload['dsm'].float()
            if dsm.shape!=(1,*mask.shape):
                raise ValueError('DSM must have shape [1,H,W] on the optical image grid.')
        else:
            dsm=torch.zeros((1,*mask.shape),dtype=image.dtype)
        return {'image':image,'dsm':dsm,'mask':mask}

class SyntheticSmokeDataset(Dataset):
    """Only for mechanical testing; never use synthetic results in manuscript."""
    def __init__(self,count=2,size=64,num_classes=5,has_dsm=True):
        self.count=count;self.size=size;self.num_classes=num_classes;self.has_dsm=has_dsm
    def __len__(self):return self.count
    def __getitem__(self,index):
        g=torch.Generator().manual_seed(index+123)
        image=torch.rand((3,self.size,self.size),generator=g)
        mask=torch.randint(self.num_classes,(self.size,self.size),generator=g)
        dsm=torch.rand((1,self.size,self.size),generator=g) if self.has_dsm else torch.zeros((1,self.size,self.size))
        return dict(image=image,mask=mask,dsm=dsm)
