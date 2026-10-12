import unittest
import torch
from configs.datasets import get_dataset
from models import FMSPNet
from checkpointing import model_weights, load_weights


class CheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def tiny_model(self):
        return FMSPNet(get_dataset('loveda'), backbone_backend='reference',
                       mamba_backend='reference', clip_backend='reference',
                       visual_dim=16, dsm_dim=8, d_state=4, reference_width=16)

    def test_full_reference_checkpoint_roundtrip(self):
        model = self.tiny_model()
        weights, adapter_only = model_weights(model)
        self.assertFalse(adapter_only)
        clone = self.tiny_model()
        load_weights(clone, {'state_dict': weights, 'adapter_only': False})
        self.assertTrue(torch.allclose(clone.lsam.text_projection.weight,
                                       model.lsam.text_projection.weight))

    def test_adapter_only_key_validation(self):
        model = self.tiny_model()
        # Model is reference ViT but checkpoint check follows SAM-style adapter
        # protocol: frozen backbone omitted, all remaining tensors required.
        payload = {'state_dict': {k: v for k,v in model.state_dict().items()
                                  if not k.startswith('backbone.')}, 'adapter_only': True}
        clone = self.tiny_model()
        load_weights(clone, payload)
        broken = {'state_dict': dict(payload['state_dict']), 'adapter_only': True}
        broken['state_dict'].pop('lsam.text_projection.weight')
        with self.assertRaises(RuntimeError):
            load_weights(clone, broken)


if __name__ == '__main__':
    unittest.main()
