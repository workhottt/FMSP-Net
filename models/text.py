from __future__ import annotations
import hashlib
import torch
from torch import nn
from torch.nn import functional as F

class FrozenCLIPText(nn.Module):
    def __init__(self, prompts: tuple[str, ...], backend: str='open_clip',
                 model_name: str='ViT-B-32', pretrained: str='openai', reference_dim: int=512):
        super().__init__()
        self.prompts = tuple(prompts)
        self.backend = backend
        if backend == 'reference':
            # Deterministic text-specific random vectors, not CLIP embeddings.
            vectors=[]
            for p in self.prompts:
                seed=int.from_bytes(hashlib.sha256(p.encode()).digest()[:8],'little') % (2**31)
                gen=torch.Generator().manual_seed(seed)
                vectors.append(torch.randn(reference_dim,generator=gen))
            self.register_buffer('embeddings', F.normalize(torch.stack(vectors),dim=-1))
        elif backend == 'open_clip':
            try:
                import open_clip
            except ImportError as exc:
                raise ImportError('Install open_clip_torch for real frozen CLIP text prototypes.') from exc
            model, _, _ = open_clip.create_model_and_transforms(model_name,pretrained=pretrained,device='cpu')
            tokenizer=open_clip.get_tokenizer(model_name)
            model.eval()
            with torch.no_grad():
                toks=tokenizer(list(self.prompts))
                emb=model.encode_text(toks).float()
                emb=F.normalize(emb,dim=-1)
            del model
            if emb.size(-1)!=512:
                raise ValueError(f'Paper specifies CLIP Ct=512; obtained {emb.size(-1)}.')
            self.register_buffer('embeddings',emb)
        else:
            raise ValueError('text backend must be open_clip or reference')
        self.out_dim=self.embeddings.size(-1)

    def forward(self)->torch.Tensor:
        return self.embeddings
