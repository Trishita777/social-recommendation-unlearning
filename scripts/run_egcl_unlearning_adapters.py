"""Versioned native-topology candidate and explicitly adapted GNNDelete for EGCL.

All deployed graphs have incident request edges removed with fixed node IDs.
Original-graph computations are auxiliary gradient/request witnesses only.
This runner does not replace or overwrite the existing EGCL controls.
"""
import argparse
import copy
import fcntl
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from run_unlearning import ROOT, atomic_json, digest, evaluate, graph, load_data, seed_everything
from run_egcl_backbone import (EGCL, atomic_torch, load_parameters, parameter_state,
                               resume_training, save_training, synchronize, train)
from social_unlearning.edge_deletion import edge_graph
from social_unlearning.egcl_unlearning_adapters import (GNNDeleteEGCL, deletion_setup,
                                                       gnndelete_loss, topology_correction)


def native_data(raw, deleted, device):
    data = graph(raw, np.arange(raw.U), device) if deleted is None else edge_graph(raw, deleted, device)
    # EGCL rebuilds native adjacency and never consumes the ODSR matrices.
    del data.adj_ui, data.adj_uu, data.norm_adj
    return data


def train_gnndelete(original, student, raw, data, config, provenance, out, source, device, seed):
    masks, fp, npair, locality = deletion_setup(raw, data.deleted_users, student.layers,
                                               seed, config['request_edge_cap'])
    model = GNNDeleteEGCL(student, masks, source)
    witness = GNNDeleteEGCL(original, masks, source, model.rec_operators, model.social_operators)
    optimizer = torch.optim.Adam(model.operator_parameters(), lr=config['deletion_lr'])
    progress = dict(epoch=0, best=-float('inf'), best_epoch=0, bad=0, complete=False,
                    optimizer_seconds=0., history=[])
    if (out/'last.pt').exists():
        progress = resume_training(out/'last.pt', model, [optimizer], config, provenance)
    with torch.no_grad():
        teacher = tuple(x.detach() for x in model.layer_states(apply=False))
    fp = torch.as_tensor(fp, device=device); npair = torch.as_tensor(npair, device=device)
    locality = [torch.as_tensor(mask, dtype=torch.bool, device=device) for mask in locality]
    while not progress['complete'] and progress['epoch'] < config['deletion_steps']:
        synchronize(device); start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss, components = gnndelete_loss(model, witness, teacher, fp, npair, locality, config['alpha'])
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite EGCL GNNDelete loss')
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.operator_parameters()):
            raise FloatingPointError('Invalid deletion operator gradient')
        optimizer.step(); progress['epoch'] += 1
        synchronize(device); progress['optimizer_seconds'] += time.perf_counter() - start
        step = progress['epoch']
        if step == 1 or step % config['deletion_validation_every'] == 0 or step == config['deletion_steps']:
            metrics, _ = evaluate(model, data, data.val, device)
            improved = metrics['recall@20'] > progress['best']
            if improved:
                progress.update(best=metrics['recall@20'], best_epoch=step, bad=0)
            else:
                progress['bad'] += 1
            row = dict(step=step, components=components, validation=metrics)
            progress['history'].append(row)
            progress['complete'] = step == config['deletion_steps'] or progress['bad'] >= config['deletion_patience']
            if improved:
                save_training(out/'best_val.pt', model, [optimizer], config, provenance, progress)
            save_training(out/'last.pt', model, [optimizer], config, provenance, progress)
            atomic_json(out/'history.json', progress['history']); print(json.dumps(row), flush=True)
    best = torch.load(out/'best_val.pt', map_location='cpu', weights_only=False)
    load_parameters(model, best['model'])
    atomic_torch(out/'deployment.pt', dict(model=parameter_state(model), config=config,
                 provenance=provenance, retained_original_ids=np.arange(raw.U),
                 adapter='GNNDeleteEGCL', graph_buffers='rebuild_from_verified_data'))
    timing = {k: v for k, v in progress.items() if k != 'history'}
    timing.update(request_edge_probe=len(fp), locality_nodes=[int(m.sum()) for m in locality])
    return model, timing


def run(args):
    torch.set_num_threads(args.cpu_threads); device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    raw = load_data(args.dataset, args.seed); split = raw.meta['split_hash'][:16]
    original_path = ROOT/'checkpoints/EGCL'/args.dataset/f'seed_{args.seed}'/split/'original/deployment.pt'
    ck = torch.load(original_path, map_location='cpu', weights_only=False)
    deleted = np.sort(np.random.default_rng(args.seed + 10000).permutation(raw.U)[:max(1, round(raw.U * args.fraction))])
    deletion = digest(deleted)
    cfg = dict(ck['config'], method=args.method, epochs=args.epochs, min_epochs=args.epochs,
               deletion_semantics='incident_edges_only_keep_user_rows',
               correction_relative_step=args.relative_step, deletion_steps=args.steps,
               deletion_lr=.001, deletion_validation_every=25, deletion_patience=10,
               request_edge_cap=4096, alpha=.5)
    source = ROOT/'external/baseline_sources/GNNDelete'
    files = [Path(__file__), ROOT/'src/social_unlearning/egcl_unlearning_adapters.py',
             ROOT/'src/social_unlearning/egcl.py', ROOT/'src/social_unlearning/edge_deletion.py',
             ROOT/'src/social_unlearning/gnndelete_edges.py', ROOT/'scripts/run_egcl_backbone.py',
             ROOT/'scripts/run_unlearning.py']
    if args.method == 'gnndelete_native':
        files += [source/'framework/models/deletion.py', source/'framework/trainer/gnndelete_nodeemb.py']
    file_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    source_hash = hashlib.sha256(json.dumps(file_hashes, sort_keys=True).encode()).hexdigest()
    prov = dict(ck['provenance'], original_source=ck['provenance']['source'], source=source_hash,
                deletion=deletion, deletion_semantics=cfg['deletion_semantics'], method=args.method,
                original_checkpoint_sha256=hashlib.sha256(original_path.read_bytes()).hexdigest())
    version = hashlib.sha256(json.dumps(dict(config=cfg, provenance=prov), sort_keys=True).encode()).hexdigest()[:16]
    out = ROOT/'checkpoints/egcl_unlearning_adapters_v1'/args.dataset/f'seed_{args.seed}'/split/deletion[:16]/version/args.method
    out.mkdir(parents=True, exist_ok=True)
    with (out/'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (out/'results.json').exists():
            report = json.loads((out/'results.json').read_text())
            if report['provenance'] != prov or report['config'] != cfg:
                raise ValueError('Completed adapter provenance mismatch')
            print('REUSE_COMPLETED', out, flush=True); return
        seed_everything(args.seed)
        resumed = (out/'last.pt').exists()
        synchronize(device)
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        data = native_data(raw, deleted, device); original_data = native_data(raw, None, device)
        original = EGCL(original_data, cfg, device); load_parameters(original, ck['model'])
        student = EGCL(data, cfg, device); load_parameters(student, ck['model'])
        protocol = dict(config=cfg, provenance=prov, source_files=file_hashes,
                        deleted_users=deleted.tolist(), counts=data.counts,
                        checkpoint_selection='retained validation recall@20 only',
                        interpretation='New native EGCL adaptations; no certificate or exact published-baseline reproduction',
                        structural_limit='All altered-graph requested native user outputs are identically zero; their score forgetting metrics are non-discriminative')
        if args.method == 'native_topology':
            protocol['algorithm'] = 'one normalized blockwise original-to-altered native gradient-difference step, deleted raw rows zeroed, then native alternating retained training'
            correction_path = out/'correction.pt'
            if correction_path.exists():
                correction = torch.load(correction_path, map_location='cpu', weights_only=False)
                if correction['provenance'] != prov:
                    raise ValueError('Correction provenance mismatch')
                load_parameters(student, correction['model'])
                correction_meta = correction['metadata']
            else:
                correction_meta = topology_correction(original, student, deleted, args.relative_step)
                atomic_torch(correction_path, dict(model=parameter_state(student), provenance=prov,
                                                   metadata=correction_meta))
            del original
            atomic_json(out/'manifest.json', protocol)
            training = train(student, data, cfg, prov, out, device)
            training['topology_correction'] = correction_meta; model = student
        else:
            protocol['algorithm'] = 'released GNNDelete DeletionLayer/nodeemb MSE form; shared operators in both EGCL streams; frozen backbone and attention'
            protocol['deviations'] = ['EGCL heat-weighted propagation and attention replace source GCN',
                                      'operators tied across recommendation/social streams',
                                      'typed UI/social negative edges and capped fixed request sample',
                                      'full original graph witnesses replace local induced graph',
                                      'random targets and locality use altered native graph',
                                      'extend layerwise nodeemb loss to all native layers and both streams',
                                      'checkpoint selected by retained recommendation validation']
            atomic_json(out/'manifest.json', protocol)
            model, training = train_gnndelete(original, student, raw, data, cfg, prov, out,
                                              source, device, args.seed)
        synchronize(device); elapsed = time.perf_counter() - start
        peak = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None
        reference_path = ROOT/'checkpoints/edge_user_v1/EGCL'/args.dataset/f'seed_{args.seed}'/split/deletion[:16]/'retrain/deployment.pt'
        reference = None
        if reference_path.exists():
            ref_ck = torch.load(reference_path, map_location='cpu', weights_only=False)
            if ref_ck['provenance'].get('deletion_semantics') != cfg['deletion_semantics'] or ref_ck['provenance']['deletion'] != deletion:
                raise ValueError('Wrong scratch reference protocol')
            reference_model = EGCL(data, cfg, device); load_parameters(reference_model, ref_ck['model'])
            _, reference = evaluate(reference_model, data, data.test, device)
        metrics, _ = evaluate(model, data, data.test, device, reference=reference)
        with torch.no_grad():
            users, _ = model()
            isolated_norm = float(users[torch.as_tensor(deleted, device=device)].norm(dim=1).max())
        if isolated_norm != 0:
            raise ValueError('Adapter reconnected isolated requested users')
        report = dict(status='completed_native_EGCL_adaptation', backbone='EGCL', method=args.method,
                      dataset=args.dataset, seed=args.seed, fraction=args.fraction, config=cfg,
                      provenance=prov, metrics=metrics, training=training,
                      requested_native_embedding_max_norm=isolated_norm,
                      efficiency=dict(wall_seconds=elapsed, peak_allocated_bytes=peak, timing_mode='shared_GPU_not_isolated',
                                      resumed=resumed, wall_scope='current invocation; excludes earlier invocation costs after resume'),
                      interpretation=protocol['interpretation'], structural_limit=protocol['structural_limit'],
                      mia_status='not_run; zero native requested rows do not establish shared-influence forgetting')
        atomic_json(out/'results.json', report)
        result = ROOT/'results/egcl_unlearning_adapters_v1'/args.dataset/f'seed_{args.seed}'/version/args.method/'results.json'
        atomic_json(result, dict(report, checkpoint=str(out/'deployment.pt')))
        print('EGCL_NATIVE_ADAPTER_RESULT', result, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--method', choices=['native_topology', 'gnndelete_native'], required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--fraction', type=float, default=.01)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--steps', type=int, default=1500)
    parser.add_argument('--relative-step', type=float, default=.001)
    parser.add_argument('--cpu-threads', type=int, default=4)
    args = parser.parse_args()
    if args.epochs < 1 or args.steps < 1 or not 0 < args.fraction < 1:
        parser.error('positive training budgets and fraction in (0,1) required')
    run(args)
