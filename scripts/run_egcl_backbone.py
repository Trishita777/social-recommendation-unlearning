"""Resumable full-objective EGCL originals and matched deletion controls, DGL only."""
import argparse
import fcntl
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from run_unlearning import ROOT, atomic_json, digest, evaluate, graph, load_data, seed_everything
from social_unlearning.egcl import EGCL


def parameter_state(model):
    # Rebuild graph buffers from the verified data, never transplant old graphs.
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters()}


def load_parameters(model, state, keep=None):
    with torch.no_grad():
        expected = dict(model.named_parameters())
        if set(state) != set(expected):
            raise ValueError('EGCL parameter schema mismatch')
        for name, p in expected.items():
            value = state[name]
            if name == 'user_embeddings' and keep is not None:
                value = value[keep]
            if value.shape != p.shape:
                raise ValueError(f'EGCL shape mismatch: {name}')
            p.copy_(value)


def atomic_torch(path, payload):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def save_training(path, model, optimizers, config, provenance, progress):
    atomic_torch(path, dict(schema=1, model=parameter_state(model),
        optimizers=[opt.state_dict() for opt in optimizers], config=config,
        provenance=provenance, progress=progress,
        rng=dict(python=random.getstate(), numpy=np.random.get_state(),
                 torch=torch.get_rng_state(),
                 cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)))


def resume_training(path, model, optimizers, config, provenance):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    if ck['config'] != config or ck['provenance'] != provenance:
        raise ValueError('EGCL resume config/provenance mismatch')
    load_parameters(model, ck['model'])
    for opt, state in zip(optimizers, ck['optimizers']):
        opt.load_state_dict(state)
    random.setstate(ck['rng']['python']); np.random.set_state(ck['rng']['numpy'])
    torch.set_rng_state(ck['rng']['torch'])
    if ck['rng']['cuda'] is not None:
        torch.cuda.set_rng_state_all(ck['rng']['cuda'])
    return ck['progress']


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def train(model, data, config, provenance, out, device):
    opts = [torch.optim.Adam(model.weight_parameters(), lr=config['learning_rate']),
            torch.optim.Adam(model.embedding_parameters(), lr=config['learning_rate'])]
    progress = dict(epoch=0, best=-1., best_epoch=0, bad=0, history=[],
                    optimizer_seconds=0., training_seconds=0., complete=False,
                    peak_allocated_bytes=0)
    if (out/'last.pt').exists():
        progress = resume_training(out/'last.pt', model, opts, config, provenance)
        print('RESUMED_EPOCH', progress['epoch'], flush=True)
    budget = config['epochs']
    while progress['epoch'] < budget and not progress['complete']:
        synchronize(device); start = time.perf_counter()
        wl, el = model.alternating_step(*opts)
        synchronize(device); opt_seconds = time.perf_counter()-start
        if not np.isfinite([wl, el]).all() or not all(torch.isfinite(p).all() for p in model.parameters()):
            raise FloatingPointError('Nonfinite EGCL objective/parameters; checkpoint not overwritten')
        ep = progress['epoch']+1
        progress['epoch'] = ep
        progress['optimizer_seconds'] += opt_seconds
        improved = False
        metrics = None
        if ep == 1 or ep % config['validation_every'] == 0 or ep == budget:
            metrics, _ = evaluate(model, data, data.val, device)
            improved = metrics['recall@20'] > progress['best']
            if improved:
                progress.update(best=metrics['recall@20'], best_epoch=ep, bad=0)
            else:
                progress['bad'] += 1
        row = dict(epoch=ep, weight_loss=wl, embedding_loss=el,
                   optimizer_seconds=opt_seconds, validation=metrics)
        progress['history'].append(row)
        progress['complete'] = ep == budget or (ep >= config['min_epochs'] and progress['bad'] >= config['patience'])
        synchronize(device)
        progress['training_seconds'] += time.perf_counter()-start
        if device.type == 'cuda':
            progress['peak_allocated_bytes'] = max(progress['peak_allocated_bytes'], torch.cuda.max_memory_allocated(device))
        checkpoint_start = time.perf_counter()
        if improved:
            save_training(out/'best_val.pt', model, opts, config, provenance, progress)
        save_training(out/'last.pt', model, opts, config, provenance, progress)
        atomic_json(out/'history.json', progress['history'])
        progress['training_seconds'] += time.perf_counter()-checkpoint_start
        print(json.dumps(row), flush=True)
    best = torch.load(out/'best_val.pt', map_location='cpu', weights_only=False)
    load_parameters(model, best['model'])
    atomic_torch(out/'deployment.pt', dict(schema=1, model=parameter_state(model),
        retained_original_ids=data.keep, config=config, provenance=provenance,
        best_epoch=progress['best_epoch'], graph_buffers='rebuild_from_verified_data'))
    return {k: v for k, v in progress.items() if k != 'history'}


def run(a):
    torch.set_num_threads(8)
    device = torch.device(a.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    raw = load_data(a.dataset, a.seed)
    keep = np.arange(raw.U)
    deletion = 'original'
    if a.method != 'original':
        deleted = np.sort(np.random.default_rng(a.seed+10000).permutation(raw.U)[:max(1, round(raw.U*a.fraction))])
        keep = np.setdiff1d(keep, deleted); deletion = digest(deleted)
    config = dict(embedding_size=64, layer_size=3, embedding_reg=2., weight_reg=60.,
        beta=.3, temperature=.05, intra_reg=1e-6, inter_reg=1e-8,
        learning_rate=.01, contrastive_chunk=256, epochs=a.epochs,
        validation_every=5, patience=20, min_epochs=100, seed=a.seed,
        method=a.method, checkpoint_selection='validation_recall@20')
    if a.method == 'retain_finetune':
        config.update(epochs=20, min_epochs=20)
    source = hashlib.sha256(b''.join(p.read_bytes() for p in [Path(__file__),
        ROOT/'src/social_unlearning/egcl.py', ROOT/'scripts/run_unlearning.py',
        ROOT/'external/EGCL/model/social_recommender/EGCL.py'])).hexdigest()
    provenance = dict(source=source, split=raw.meta['split_hash'], social=raw.meta['social_hash'],
        data=raw.meta['source_files'], deletion=deletion,
        backbone='EGCL_equation_port', social_policy='directed adjacency plus transpose; no added self loops')
    base = ROOT/'checkpoints/EGCL'/a.dataset/f'seed_{a.seed}'/raw.meta['split_hash'][:16]
    out = base/'original' if a.method == 'original' else base/deletion[:16]/a.method
    if a.pilot:
        out = ROOT/'checkpoints/egcl_pilot'/a.dataset/f'seed_{a.seed}'
        config.update(epochs=2, validation_every=1, min_epochs=2)
    out.mkdir(parents=True, exist_ok=True)
    with (out/'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (out/'results.json').exists() and (out/'deployment.pt').exists():
            report = json.loads((out/'results.json').read_text())
            if report['config'] != config or report['provenance'] != provenance:
                raise ValueError('Completed EGCL cache provenance mismatch')
            print('REUSE_COMPLETED', out, flush=True); return
        seed_everything(a.seed)
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        data = graph(raw, keep, device)
        # EGCL reconstructs its own native adjacency from ui/social.
        del data.adj_ui, data.adj_uu, data.norm_adj
        model = EGCL(data, config, device)
        graph_seconds = time.perf_counter()-start
        if a.method in ('retain_finetune', 'graph_only'):
            original = torch.load(base/'original/deployment.pt', map_location='cpu', weights_only=False)
            if original['provenance']['source'] != source or original['provenance']['split'] != provenance['split']:
                raise ValueError('Original EGCL source/split mismatch')
            load_parameters(model, original['model'], keep)
        atomic_json(out/'data_manifest.json', dict(**raw.meta, config=config,
            egcl_social_policy=provenance['social_policy'], retained_counts=data.counts))
        if a.method == 'graph_only':
            atomic_torch(out/'deployment.pt', dict(schema=1, model=parameter_state(model), retained_original_ids=keep,
                config=config, provenance=provenance, best_epoch=0, graph_buffers='rebuild_from_verified_data'))
            timing = dict(epoch=0, optimizer_seconds=0., training_seconds=0., complete=True)
        else:
            timing = train(model, data, config, provenance, out, device)
        synchronize(device)
        wall = time.perf_counter()-start
        peak = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0
        metrics, _ = evaluate(model, data, data.test, device)
        report = dict(status='pilot_only' if a.pilot else 'completed', backbone='EGCL', method=a.method,
            dataset=a.dataset, seed=a.seed, fraction=0 if a.method == 'original' else a.fraction,
            config=config, provenance=provenance, metrics=metrics, counts=data.counts,
            timing=timing, efficiency=dict(process_wall_seconds=wall, graph_seconds=graph_seconds,
                peak_allocated_bytes=peak, timing_mode='shared_gpu_training_not_isolated_efficiency'),
            limitations='Original training and controls only; EGCL proposed/published-baseline adapters and EGCL-specific shadow MIA are pending. Original config from official conf/EGCL.properties, with validation early stopping and common study splits. Epoch cap is not a claim of convergence.')
        atomic_json(out/'results.json', report)
        print('EGCL_RESULT', out/'results.json', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['epinions', 'yelp2018'], required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--method', choices=['original', 'retrain', 'retain_finetune', 'graph_only'], default='original')
    p.add_argument('--fraction', type=float, default=.01)
    p.add_argument('--epochs', type=int, default=2000)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--pilot', action='store_true')
    run(p.parse_args())
