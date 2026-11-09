"""Atomic, provenance-checked research checkpoints. Load trusted project files only."""
import os
import random
import tempfile
from pathlib import Path
import numpy as np
import torch


def save_checkpoint(path, model, optimizer, *, config, provenance, epoch, step,
                    metrics=None, complete=False, scheduler=None):
    if not provenance or not all(provenance.values()):
        raise ValueError('Nonempty provenance is required')
    payload = dict(schema=1, model=model.state_dict(),
                   optimizer=optimizer.state_dict() if optimizer is not None else None,
                   scheduler=scheduler.state_dict() if scheduler is not None else None,
                   config=config, provenance=provenance, epoch=epoch, step=step,
                   metrics=metrics or {}, complete=complete,
                   rng=dict(python=random.getstate(), numpy=np.random.get_state(),
                            torch=torch.get_rng_state(),
                            cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + path.name, suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def load_checkpoint(path, model, optimizer, *, expected_provenance, expected_config,
                    scheduler=None, restore_rng=True):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    if ckpt.get('schema') != 1:
        raise ValueError('Unsupported checkpoint schema')
    if ckpt['provenance'] != expected_provenance:
        raise ValueError('Checkpoint provenance mismatch; refusing resume')
    if ckpt['config'] != expected_config:
        raise ValueError('Checkpoint config mismatch; refusing resume')
    model.load_state_dict(ckpt['model'])
    if optimizer is not None:
        if ckpt['optimizer'] is None:
            raise ValueError('Missing optimizer state')
        optimizer.load_state_dict(ckpt['optimizer'])
    if scheduler is not None:
        if ckpt['scheduler'] is None:
            raise ValueError('Missing scheduler state')
        scheduler.load_state_dict(ckpt['scheduler'])
    if restore_rng:
        r = ckpt['rng']
        random.setstate(r['python'])
        np.random.set_state(r['numpy'])
        torch.set_rng_state(r['torch'])
        if r['cuda'] is not None:
            if len(r['cuda']) != torch.cuda.device_count():
                raise ValueError('CUDA device count differs; cannot exactly resume RNG')
            torch.cuda.set_rng_state_all(r['cuda'])
    return ckpt
