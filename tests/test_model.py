import unittest
import tempfile
from pathlib import Path
import torch
from configs.datasets import get_dataset
from models import FMSPNet,JointLoss
from metrics import confusion_update,scores
from inference import infer_tensor,load_model

class TestFullPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def make_model(self,dataset='potsdam'):
        cfg=get_dataset(dataset)
        model=FMSPNet(cfg,backbone_backend='reference',mamba_backend='reference',
                      clip_backend='reference',visual_dim=16,dsm_dim=8,d_state=4,reference_width=16)
        return model,cfg

    def test_fsa_complement_and_gradients(self):
        from models.fsa import FSA
        fs=FSA(8,mask_grid=8)
        mask=fs.frequency_mask(4,4)
        self.assertTrue(bool(((mask>=0)&(mask<=1)).all()))
        x=torch.randn(1,8,4,4,requires_grad=True)
        fs(x).sum().backward()
        self.assertIsNotNone(fs.mask_logits.grad)

    def test_potsdam_forward_backward_and_memory(self):
        model,cfg=self.make_model('potsdam')
        model.train()
        x=torch.rand(1,3,32,32)
        dsm=torch.rand(1,1,32,32)
        y=torch.randint(len(cfg.classes),(1,32,32))
        result=model(x,dsm,update_memory=True)
        self.assertEqual(tuple(result['logits'].shape),(1,5,32,32))
        self.assertTrue(result['dsm_active'])
        self.assertTrue(bool(model.dpr.initialized.any()))
        loss,_=JointLoss()(result,y,model)
        loss.backward()
        self.assertTrue(any(p.grad is not None for p in model.fsas.parameters()))

    def test_single_modality_missing_dsm_is_zero(self):
        model,cfg=self.make_model('loveda')
        model.eval()
        x=torch.rand(1,3,32,32)
        with torch.no_grad():
            a=model(x,None)
            b=model(x,torch.ones(1,1,32,32))
        self.assertFalse(a['dsm_active'])
        self.assertTrue(torch.allclose(a['logits'],b['logits']))

    def test_checkpoint_and_inference(self):
        model,cfg=self.make_model('uavid')
        model.eval()
        img=torch.rand(3,32,32)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'model.pt'
            torch.save({'state_dict':model.state_dict(),'dataset':'uavid',
                        'model_settings':model.model_settings,'is_smoke_test':True},path)
            loaded,_,state=load_model(str(path))
            result=infer_tensor(loaded,{'image':img})
            self.assertEqual(tuple(result['prediction'].shape),(1,32,32))
            self.assertEqual(tuple(result['activation'].shape),(1,8,2,2))
            self.assertTrue(state['is_smoke_test'])

    def test_confusion_metrics(self):
        hist=torch.zeros((3,3),dtype=torch.int64)
        logits=torch.randn(1,3,8,8)
        target=torch.randint(3,(1,8,8))
        report=scores(confusion_update(hist,logits,target))
        self.assertTrue(all(k in report for k in ('mIoU','mF1','OA')))

if __name__=='__main__':unittest.main()
