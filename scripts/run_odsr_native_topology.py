"""New versioned ODSR native-topology/reset/retention candidate and ablations.

Hyperparameters are fixed before test evaluation. This is a finite-probe,
first-order empirical candidate, not certified unlearning or the old 'ours'.
"""
import argparse
import fcntl
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from run_unlearning import (ROOT, ODSR, atomic_json, compact, digest, evaluate, graph,
                            load_data, loss_terms, negatives, seed_everything)
from run_egcl_backbone import atomic_torch, synchronize
from audit_corrected_forgetting import compare, probes, score_probe
from social_unlearning.benchmark_mio import BenchmarkMIO, embedding_features
from social_unlearning.checkpoint import load_checkpoint, save_checkpoint
from social_unlearning.edge_deletion import edge_graph, edge_sensitivity
from social_unlearning.isolated_reset import reset_isolated_odsr_rows
from social_unlearning.odsr_native_topology import paired_probe, topology_correction

VARIANTS = ['full', 'no_topology', 'uniform_weights', 'no_reset']


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def prepare(original, student, original_data, data, raw, cfg, seed, variant):
    probe = paired_probe(original_data, data.deleted_users, seed + 610000, cfg['probe_size'])
    if variant == 'no_topology':
        correction = dict(status='ablated', original_triplets=len(probe['pairs']),
                          retained_triplets=int(probe['retained'].sum()), probe_seed=probe['seed'])
    else:
        correction = topology_correction(original, student, data.deleted_users, probe,
                                          cfg['correction_relative_step'])
    if variant == 'no_reset':
        reset = dict(status='ablated')
    else:
        reset = reset_isolated_odsr_rows(student, data, data.deleted_users, seed + 710000)
    weights = edge_sensitivity(original, student, data, seed, mix=cfg['weight_mix'])
    if variant == 'uniform_weights':
        weights = {name: torch.ones_like(value) for name, value in weights.items()}
    # The native retained objective cannot reach isolated requested rows. Guard
    # that property structurally throughout training rather than comparing GPUs.
    metadata = dict(correction=correction, reset=reset, variant=variant,
                    sensitivity='computed once after correction/reset from immutable original and altered student',
                    weight_mode='uniform' if variant == 'uniform_weights' else 'normalized sensitivity with mix .25',
                    probe_hash=digest(np.column_stack([probe['pairs'], probe['negative']])))
    return weights, metadata


def train_retained(model, data, weights, cfg, provenance, out, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    rng = np.random.default_rng(cfg['seed'])
    progress = dict(epoch=0, best=-1., best_epoch=0, steps=0, optimizer_seconds=0., history=[], complete=False)
    if (out/'last.pt').exists():
        ck = load_checkpoint(out/'last.pt', model, optimizer,
                             expected_provenance=provenance, expected_config=cfg)
        progress = ck['metrics']['progress']; rng.bit_generator.state = ck['metrics']['numpy_generator']
    deleted = torch.as_tensor(data.deleted_users, device=device)
    isolated_initial = model.embedding_dict['user_emb'][deleted].detach().clone()
    while progress['epoch'] < cfg['epochs'] and not progress['complete']:
        model.train(); order = rng.permutation(len(data.train)); total = 0.
        synchronize(device); start = time.perf_counter()
        for offset in range(0, len(order), cfg['batch_size']):
            ids = order[offset:offset + cfg['batch_size']]; pairs = data.train[ids]
            u, p = torch.as_tensor(pairs, device=device).T
            n = torch.as_tensor(negatives(data, pairs[:, 0], rng), device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_terms(model, u, p, n, weights=weights,
                              indices=torch.as_tensor(ids, device=device), mode='weighted_finetune')
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite weighted native retained loss')
            loss.backward(); optimizer.step()
            progress['steps'] += 1; total += float(loss.detach()) * len(ids)
        synchronize(device); progress['optimizer_seconds'] += time.perf_counter() - start
        if not torch.equal(isolated_initial, model.embedding_dict['user_emb'][deleted]):
            raise AssertionError('Native retained training changed isolated requested rows')
        progress['epoch'] += 1
        metrics, _ = evaluate(model, data, data.val, device)
        if metrics['recall@20'] > progress['best']:
            progress.update(best=metrics['recall@20'], best_epoch=progress['epoch'])
            atomic_torch(out/'best_val.pt', dict(model=model.state_dict(), provenance=provenance))
        progress['complete'] = progress['epoch'] >= cfg['epochs']
        row = dict(epoch=progress['epoch'], native_weighted_loss=total/len(order), validation=metrics)
        progress['history'].append(row)
        save_checkpoint(out/'last.pt', model, optimizer, config=cfg, provenance=provenance,
                        epoch=progress['epoch'], step=progress['steps'], complete=progress['complete'],
                        metrics=dict(progress=progress, numpy_generator=rng.bit_generator.state))
        atomic_json(out/'history.json', progress['history']); print(json.dumps(row), flush=True)
    best = torch.load(out/'best_val.pt', map_location=device, weights_only=False)
    if best['provenance'] != provenance:
        raise ValueError('Best native checkpoint provenance mismatch')
    model.load_state_dict(best['model'])
    return {k: v for k, v in progress.items() if k != 'history'}


def optional_mio(bank, raw, deleted, embeddings):
    if bank is None:
        return dict(status='not_requested')
    manifest = json.loads((bank/'manifest.json').read_text())
    if manifest['provenance']['backbone'] != 'ODSR':
        raise ValueError('Expected an ODSR shadow bank')
    shadow_raw = load_data(raw.meta['dataset'], 42)
    if manifest['provenance']['split'] != shadow_raw.meta['split_hash'] or raw.um != shadow_raw.um or raw.im != shadow_raw.im:
        raise ValueError('Frozen shadow dataset/split/ID mapping mismatch')
    forbidden = np.concatenate([np.asarray(ids) for ids in manifest['roles'].values()]
                               + [np.asarray(manifest['shadow_member_users'])])
    if np.intersect1d(deleted, forbidden).size or not np.isin(deleted, manifest['excluded_target_users']).all():
        return dict(status='unavailable_identity_overlap',
                    explanation='This request is not disjoint from frozen shadow/attack identities; no score reported')
    ck = torch.load(bank/'attacker.pt', map_location='cpu', weights_only=False)
    if ck['provenance'] != manifest['provenance']:
        raise ValueError('Frozen attacker and manifest provenance differ')
    known = defaultdict(list)
    for user, item in np.concatenate([raw.train, raw.val]):
        known[int(user)].append(int(item))
    users, items = embeddings
    features = embedding_features(users.cpu(), items.cpu(), deleted,
                                  {user: user for user in range(raw.U)}, known)
    attacker = BenchmarkMIO(features.shape[1]); attacker.load_state_dict(ck['model'])
    scores = attacker.member_probability(features).numpy()
    return dict(status='positive_only_frozen_benchmark_MIO', mean_probability=float(scores.mean()),
                member_classification_rate=float((scores >= .5).mean()), attacker=str(bank/'attacker.pt'),
                attacker_sha256=file_hash(bank/'attacker.pt'),
                interpretation='No genuine target nonmembers; not target accuracy/AUC or a privacy guarantee')


def run(args):
    torch.set_num_threads(args.cpu_threads); device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    raw = load_data(args.dataset, args.seed); split = raw.meta['split_hash'][:16]
    original_path = ROOT/'checkpoints/ODSR'/args.dataset/f'seed_{args.seed}'/split/'original/deployment.pt'
    ck = torch.load(original_path, map_location='cpu', weights_only=False)
    deleted = np.sort(np.random.default_rng(args.seed + 10000).permutation(raw.U)[:max(1, round(raw.U * args.fraction))])
    deletion = digest(deleted)
    cfg = dict(ck['config'], seed=args.seed, lr=.0001, epochs=5, weight_mix=.25,
               correction_relative_step=.001, probe_size=1024, variant=args.variant,
               deletion_semantics='incident_edges_only_keep_user_rows',
               method=f'native_topology_{args.variant}', checkpoint_selection='retained_validation_recall@20')
    files = [Path(__file__), ROOT/'scripts/run_unlearning.py', ROOT/'scripts/run_egcl_backbone.py',
             ROOT/'scripts/audit_corrected_forgetting.py', ROOT/'external/ODSR/ODSR.py']
    files += [ROOT/'src/social_unlearning'/name for name in
              ['odsr_native_topology.py', 'edge_deletion.py', 'isolated_reset.py', 'checkpoint.py',
               'native_guarded.py', 'native_guarded_memory.py', 'benchmark_mio.py']]
    hashes = {str(path.relative_to(ROOT)): file_hash(path) for path in files}
    provenance = dict(ck['provenance'], source=hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
                      original_source=ck['provenance']['source'], original_checkpoint_sha256=file_hash(original_path),
                      deletion=deletion, deletion_semantics=cfg['deletion_semantics'], variant=args.variant)
    version = hashlib.sha256(json.dumps(dict(config=cfg, provenance=provenance), sort_keys=True).encode()).hexdigest()[:16]
    suffix = Path('ODSR')/args.dataset/f'seed_{args.seed}'/split/deletion[:16]/version/args.variant
    out = ROOT/'checkpoints/odsr_native_topology_v1'/suffix; out.mkdir(parents=True, exist_ok=True)
    with (out/'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (out/'results.json').exists():
            report = json.loads((out/'results.json').read_text())
            if report['provenance'] != provenance or report['config'] != cfg:
                raise ValueError('Completed native topology result provenance mismatch')
            print('REUSE_COMPLETED', out, flush=True); return
        seed_everything(args.seed); synchronize(device)
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        resumed = (out/'last.pt').exists(); start = time.perf_counter()
        data = edge_graph(raw, deleted, device)
        original_data = graph(raw, np.arange(raw.U), device)
        original = ODSR(original_data, cfg, device).to(device); original.load_state_dict(ck['model'])
        model = compact(original, data, cfg, device)
        preparation_path = out/'preparation.pt'
        if preparation_path.exists():
            preparation = torch.load(preparation_path, map_location=device, weights_only=False)
            if preparation['provenance'] != provenance:
                raise ValueError('Preparation provenance mismatch')
            model.load_state_dict(preparation['model']); weights = preparation['weights']; metadata = preparation['metadata']
        else:
            weights, metadata = prepare(original, model, original_data, data, raw, cfg, args.seed, args.variant)
            atomic_torch(preparation_path, dict(model=model.state_dict(), weights=weights,
                                                metadata=metadata, provenance=provenance))
        del original, original_data
        atomic_json(out/'manifest.json', dict(config=cfg, provenance=provenance, source_files=hashes,
                    deleted_users=deleted.tolist(), fraction=args.fraction, counts=data.counts,
                    preparation=metadata, objective='weighted native retained BPR+InfoNCE+native regularization only',
                    no_test_selection=True,
                    limitation='Finite eval-mode probe and first-order normalized correction; no guarantee of shared-influence erasure. Ablations are separate fixed-budget variants.'))
        training = train_retained(model, data, weights, cfg, provenance, out, device)
        atomic_torch(out/'deployment.pt', dict(model=model.state_dict(), config=cfg, provenance=provenance,
                     training=training, retained_original_ids=np.arange(raw.U), deleted_user_ids=deleted))
        synchronize(device); elapsed = time.perf_counter() - start
        peak = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None
        reference_path = ROOT/'checkpoints/edge_user_v1/ODSR'/args.dataset/f'seed_{args.seed}'/split/deletion[:16]/'retrain/deployment.pt'
        reference = None; reference_provenance = None
        if reference_path.exists():
            ref = torch.load(reference_path, map_location='cpu', weights_only=False)
            if ref['provenance'].get('deletion_semantics') != cfg['deletion_semantics'] or ref['provenance']['deletion'] != deletion:
                raise ValueError('Wrong scratch reference')
            reference_model = ODSR(data, ref['config'], device).to(device); reference_model.load_state_dict(ref['model'])
            _, reference = evaluate(reference_model, data, data.test, device)
            reference_provenance = ref['provenance']
        metrics, embeddings = evaluate(model, data, data.test, device, reference=reference)
        probe = probes(raw, deleted, args.seed)
        records = score_probe((embeddings[0].cpu(), embeddings[1].cpu()), probe)
        forgetting = {name: float(np.mean([row[name] for row in records])) for name in
                      ['positive_beats_negative', 'mean_pair_margin', 'mean_pair_bpr_loss', 'mean_positive_score']}
        if reference is not None:
            ref_records = score_probe((reference[0].cpu(), reference[1].cpu()), probe)
            forgetting.update(compare(records, ref_records))
            forgetting['scratch_positive_beats_negative'] = float(np.mean([r['positive_beats_negative'] for r in ref_records]))
        mio = optional_mio(args.bank.resolve() if args.bank else None, raw, deleted, embeddings[:2])
        report = dict(status='completed_new_native_topology_candidate', backbone='ODSR', dataset=args.dataset,
                      seed=args.seed, fraction=args.fraction, variant=args.variant, method=cfg['method'],
                      config=cfg, provenance=provenance, metrics=metrics, training=training, preparation=metadata,
                      forgetting=forgetting, reference_provenance=reference_provenance,
                      reference_status='matched_corrected_scratch' if reference is not None else 'pending_matching_fraction_scratch',
                      mio=mio, efficiency=dict(wall_seconds=elapsed, peak_allocated_bytes=peak,
                          resumed=resumed, timing_mode='shared_GPU_not_isolated',
                          wall_scope='current invocation; earlier invocation costs excluded after resume'),
                      limitation='Not old reverse-BPR ours. Finite-probe first-order adaptation; direct forgetting is not MIA or a certificate.')
        atomic_json(out/'results.json', report)
        result = ROOT/'results/odsr_native_topology_v1'/suffix/'results.json'
        atomic_json(result, dict(report, checkpoint=str(out/'deployment.pt')))
        print('ODSR_NATIVE_TOPOLOGY_RESULT', result, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--variant', choices=VARIANTS, default='full')
    parser.add_argument('--fraction', type=float, default=.01)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--bank', type=Path)
    parser.add_argument('--cpu-threads', type=int, default=4)
    args = parser.parse_args()
    if not 0 < args.fraction < 1:
        parser.error('fraction must lie in (0,1)')
    run(args)
