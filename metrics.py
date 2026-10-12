import torch

@torch.no_grad()
def confusion_update(hist,logits,target,ignore_index=255):
    pred=logits.argmax(dim=1)
    n=hist.shape[0]
    valid=(target!=ignore_index)&(target>=0)&(target<n)
    if not valid.any(): return hist
    pairs=(target[valid]*n+pred[valid]).long()
    return hist+torch.bincount(pairs,minlength=n*n).reshape(n,n).to(hist.device)

@torch.no_grad()
def scores(confusion):
    tp=confusion.diagonal().double()
    gt=confusion.sum(dim=1).double()
    pred=confusion.sum(dim=0).double()
    denom=gt+pred-tp
    iou=torch.where(denom>0,tp/denom,torch.nan)
    f1=torch.where(gt+pred>0,2*tp/(gt+pred),torch.nan)
    return {
        'mIoU':float(torch.nanmean(iou).item()*100),
        'mF1':float(torch.nanmean(f1).item()*100),
        'OA':float(tp.sum().item()/confusion.sum().clamp(min=1).item()*100),
        'per_class_IoU':[float(x*100) for x in iou],
        'per_class_F1':[float(x*100) for x in f1],
    }
