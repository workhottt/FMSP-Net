from __future__ import annotations
from pathlib import Path
import torch


def model_weights(model):
    state = model.state_dict()
    is_adapter_only = model.model_settings['backbone_backend'] == 'sam'
    if is_adapter_only:
        state = {key: value for key, value in state.items() if not key.startswith('backbone.')}
    return state, is_adapter_only


def load_weights(model, state):
    missing, unexpected = model.load_state_dict(state['state_dict'], strict=False)
    if unexpected:
        raise RuntimeError(f'Unexpected checkpoint tensors: {unexpected}')
    if state.get('adapter_only', False):
        bad_missing = [key for key in missing if not key.startswith('backbone.')]
        if bad_missing:
            raise RuntimeError(f'Adapter-only checkpoint is incomplete: {bad_missing}')
        if not missing:
            raise RuntimeError('Adapter-only checkpoint expected to omit frozen SAM backbone weights.')
    elif missing:
        raise RuntimeError(f'Full checkpoint is missing weights: {missing}')
    return model


def read_checkpoint(path, map_location=''):
    return torch.load(Path(path), map_location=map_location, weights_only=False)
