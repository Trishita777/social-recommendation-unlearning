"""Separate ODSR targets with genuine never-members; frozen CURE4Rec-style MIO.

No full-population utility model can be reused as the controlled original.
The existing independent shadow/attacker can be reused: both audit classes are
sampled only from its excluded identities. Labels always mean ORIGINAL target
membership, including after deletion. All native row dimensions remain fixed.
"""
import argparse
import copy
import fcntl
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from run_unlearning import (ROOT, ODSR, atomic_json, compact, digest, evaluate,
                            load_data, loss_terms, negatives, seed_everything)
from run_egcl_backbone import atomic_torch
from run_edge_user_odsr import optimize
from social_unlearning.checkpoint import load_checkpoint, save_checkpoint
from social_unlearning.edge_deletion import edge_graph, filtered_raw
from social_unlearning.benchmark_mio import BenchmarkMIO, embedding_features
from social_unlearning.isolated_reset import reset_isolated_odsr_rows
from social_unlearning.controlled_mio import (attack_metrics, bootstrap_attack,
                                              target_partition, validate_graph_membership)

METHODS = ['original', 'retrain', 'graph_only', 'retain_finetune', 'ours',
           'reset_graph_only', 'reset_finetune', 'scif', 'gnndelete', 'da2']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def native_train(model, data, cfg, out, provenance, epochs, device):
    """Same native objective/selection, resumable with only last/best checkpoints."""
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    rng = np.random.default_rng(cfg['seed'])
    progress = dict(epoch=0, best=-1., best_epoch=0, bad=0, steps=0,
                    optimizer_seconds=0., history=[], complete=False)
    if (out/'last.pt').exists():
        ck = load_checkpoint(out/'last.pt', model, optimizer,
                             expected_provenance=provenance, expected_config=cfg)
        progress = ck['metrics']['progress']
        rng.bit_generator.state = ck['metrics']['numpy_generator']
    while progress['epoch'] < epochs and not progress['complete']:
        model.train(); order = rng.permutation(len(data.train)); total = 0.
        synchronize(device); start = time.perf_counter()
        for offset in range(0, len(order), cfg['batch_size']):
            ids = order[offset:offset + cfg['batch_size']]; pairs = data.train[ids]
            u, p = torch.as_tensor(pairs, device=device).T
            n = torch.as_tensor(negatives(data, pairs[:, 0], rng), device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_terms(model, u, p, n, mode='retain_finetune')
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite controlled target native loss')
            loss.backward(); optimizer.step(); progress['steps'] += 1
            total += float(loss.detach()) * len(ids)
        synchronize(device); progress['optimizer_seconds'] += time.perf_counter() - start
        progress['epoch'] += 1
        metrics, _ = evaluate(model, data, data.val, device)
        improved = metrics['recall@20'] > progress['best']
        if improved:
            progress.update(best=metrics['recall@20'], best_epoch=progress['epoch'], bad=0)
            atomic_torch(out/'best_native.pt', dict(model=model.state_dict(), provenance=provenance))
        else:
            progress['bad'] += 1
        progress['complete'] = progress['epoch'] >= epochs or progress['bad'] >= cfg['patience']
        row = dict(epoch=progress['epoch'], loss=total/len(order), validation=metrics)
        progress['history'].append(row)
        save_checkpoint(out/'last.pt', model, optimizer, config=cfg, provenance=provenance,
                        epoch=progress['epoch'], step=progress['steps'], complete=progress['complete'],
                        metrics=dict(progress=progress, numpy_generator=rng.bit_generator.state))
        atomic_json(out/'history.json', progress['history']); print(json.dumps(row), flush=True)
    best = torch.load(out/'best_native.pt', map_location=device, weights_only=False)
    if best['provenance'] != provenance:
        raise ValueError('Best native checkpoint provenance mismatch')
    model.load_state_dict(best['model'])
    return {k: v for k, v in progress.items() if k != 'history'}


def make_graphs(raw, deleted, nonmembers, device):
    before = edge_graph(raw, nonmembers, device)
    after = edge_graph(raw, np.union1d(deleted, nonmembers), device)
    # Absent users are not deletion requests or retained contrastive candidates.
    before.deleted_users = np.array([], dtype=np.int64)
    before.retained_users = np.setdiff1d(np.arange(raw.U), nonmembers)
    after.deleted_users = np.asarray(deleted)
    after.retained_users = np.setdiff1d(np.arange(raw.U), np.union1d(deleted, nonmembers))
    validate_graph_membership(before, deleted, nonmembers, before_deletion=True)
    validate_graph_membership(after, deleted, nonmembers, before_deletion=False)
    return before, after


def load_model(out, data, raw_members, deleted, device):
    ck = torch.load(out/'deployment.pt', map_location='cpu', weights_only=False)
    model = ODSR(data, ck['config'], device).to(device)
    if out.name == 'gnndelete':
        from social_unlearning.gnndelete_edges import GNNDeleteODSR, neighborhoods
        masks, _, _ = neighborhoods(raw_members, deleted, model.n_layers)
        model = GNNDeleteODSR(model, masks, ROOT/'external/baseline_sources/GNNDelete')
    model.load_state_dict(ck['model']); model.eval()
    return model, ck


def evaluate_completed(base, result_dir, raw, raw_members, before, after, deleted,
                       nonmembers, bank, provenance, device, bootstrap_samples):
    saved = torch.load(bank/'attacker.pt', map_location='cpu', weights_only=False)
    histories = defaultdict(list)
    for u, i in np.concatenate([raw.train, raw.val]):
        histories[int(u)].append(int(i))
    queries = np.concatenate([deleted, nonmembers]); labels = np.concatenate([
        np.ones(len(deleted), dtype=np.int64), np.zeros(len(nonmembers), dtype=np.int64)])
    threshold = .5; threshold_source = 'fixed_0.5'
    diagnostics = bank/'diagnostics.json'
    if diagnostics.exists():
        report = json.loads(diagnostics.read_text())
        if 'validation_selected_threshold' in report:
            threshold = report['validation_selected_threshold']
            threshold_source = dict(path=str(diagnostics), sha256=sha(diagnostics),
                                    selection='existing shadow validation only')
    rows = []; probabilities = {}; utility = {}
    for method in METHODS:
        out = base/method
        if not (out/'completion.json').exists():
            continue
        completion = json.loads((out/'completion.json').read_text())
        if completion['provenance'] != dict(provenance, method=method):
            raise ValueError('Controlled target completion provenance mismatch')
        data = before if method == 'original' else after
        model, ck = load_model(out, data, raw_members, deleted, device)
        if ck['provenance'] != completion['provenance']:
            raise ValueError('Controlled target deployment provenance mismatch')
        with torch.no_grad():
            users, items = model()
            x = embedding_features(users.cpu(), items.cpu(), queries,
                                   {u: u for u in range(raw.U)}, histories)
        attacker = BenchmarkMIO(x.shape[1]); attacker.load_state_dict(saved['model'])
        score = attacker.member_probability(x).numpy(); probabilities[method] = score
        # Identical retained-user utility query set and candidate exclusion across methods.
        utility[method], _ = evaluate(model, after, after.test, device)
        row = dict(method=method, metrics_at_0_5=attack_metrics(labels, score),
                   metrics_at_shadow_validation_threshold=attack_metrics(labels, score, threshold),
                   uncertainty=bootstrap_attack(labels, score, threshold,
                                               seed=raw.meta['seed'], samples=bootstrap_samples),
                   retained_utility=utility[method], training=completion['training'],
                   efficiency=completion['efficiency'])
        rows.append(row)
        del model, users, items, x, attacker
    if 'retrain' in probabilities:
        for row in rows:
            gap = probabilities[row['method']] - probabilities['retrain']
            row['paired_mean_probability_gap_to_retrain'] = dict(
                deleted_members=float(gap[:len(deleted)].mean()),
                never_members=float(gap[len(deleted):].mean()))
    report = dict(status='controlled_target_audit', backbone='ODSR', dataset=raw.meta['dataset'],
                  seed=raw.meta['seed'], provenance=provenance, rows=rows,
                  threshold_source=threshold_source,
                  pending_methods=[m for m in METHODS if m not in probabilities],
                  query_users=queries.tolist(), original_membership_labels=labels.tolist(),
                  probabilities={k: v.tolist() for k, v in probabilities.items()},
                  feature_schema='concat(native propagated user, mean native item of known original train+validation history)',
                  threat_model='representation access and known history; independent frozen shadow attacker',
                  interpretation='Separate controlled target suite, not the full-population utility targets. Labels denote original membership. Read original target attack strength before post-deletion results; weak transfer is inconclusive. Bootstrap conditions on one target and one attacker; no low-FPR claim.')
    atomic_json(result_dir/'results.json', report)
    print('CONTROLLED_MIO_RESULT', result_dir/'results.json', flush=True)


def run(args):
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    bank = args.bank.resolve(); manifest = json.loads((bank/'manifest.json').read_text())
    attacker_ck = torch.load(bank/'attacker.pt', map_location='cpu', weights_only=False)
    if attacker_ck['provenance'] != manifest['provenance']:
        raise ValueError('Frozen attacker does not match the shadow manifest')
    del attacker_ck
    if manifest['provenance']['backbone'] != 'ODSR':
        raise ValueError('This pilot supports only ODSR')
    if manifest['config'].get('seed') != 1042:
        raise ValueError('Expected the existing independent shadow seed1042')
    raw = load_data(args.dataset, args.seed)
    shadow_raw = load_data(args.dataset, 42)
    if manifest['provenance']['split'] != shadow_raw.meta['split_hash']:
        raise ValueError('Wrong dataset/split for the shadow bank')
    if raw.um != shadow_raw.um or raw.im != shadow_raw.im:
        raise ValueError('Shadow/target ID mapping differs')
    deleted, nonmembers = target_partition(raw.U, manifest, args.seed, args.fraction)
    config_path = ROOT/'checkpoints/ODSR'/args.dataset/f'seed_{args.seed}'/raw.meta['split_hash'][:16]/'original/deployment.pt'
    config_ck = torch.load(config_path, map_location='cpu', weights_only=False)
    cfg = dict(config_ck['config'], seed=args.seed,
               deletion_semantics='incident_edges_only_keep_user_rows', controlled_target_mio=True)
    source_paths = [Path(__file__), ROOT/'src/social_unlearning/controlled_mio.py',
                    ROOT/'scripts/run_unlearning.py', ROOT/'scripts/run_edge_user_odsr.py',
                    ROOT/'external/ODSR/ODSR.py', ROOT/'src/social_unlearning/checkpoint.py',
                    ROOT/'src/social_unlearning/benchmark_mio.py', ROOT/'src/social_unlearning/isolated_reset.py',
                    ROOT/'scripts/run_egcl_backbone.py']
    source_paths += [ROOT/'src/social_unlearning'/name for name in
                     ['edge_deletion.py', 'native_guarded.py', 'native_guarded_memory.py',
                      'baselines.py', 'scif_edges.py', 'gnndelete_edges.py']]
    sources = {str(p.relative_to(ROOT)): sha(p) for p in source_paths}
    provenance = dict(protocol='controlled_target_mio_v1', split=raw.meta['split_hash'],
                      source=hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest(),
                      data=config_ck['provenance']['data'], social=raw.meta['social_hash'],
                      shadow_manifest=sha(bank/'manifest.json'), attacker=sha(bank/'attacker.pt'),
                      deleted=digest(deleted), never_members=digest(nonmembers),
                      config=hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
                      deletion_semantics=cfg['deletion_semantics'])
    experiment = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()[:16]
    suffix = Path('ODSR')/args.dataset/f'seed_{args.seed}'/experiment
    base = ROOT/'checkpoints/controlled_target_mio'/suffix
    result_dir = ROOT/'results/controlled_target_mio'/suffix
    base.mkdir(parents=True, exist_ok=True)
    suite_lock = (base/'suite.lock').open('w')
    fcntl.flock(suite_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Preparing checks graph membership on CPU and never allocates CUDA.
    before, after = make_graphs(raw, deleted, nonmembers, 'cpu' if args.prepare_only else device)
    raw_members = filtered_raw(raw, nonmembers)
    protocol = dict(provenance=provenance, source_files=sources, config=cfg,
                    frozen_attacker=str(bank/'attacker.pt'), requested_users=deleted.tolist(),
                    never_member_users=nonmembers.tolist(), full_users=raw.U,
                    requested_fraction_of_full_universe=args.fraction,
                    requested_fraction_of_original_members=len(deleted)/(raw.U-len(nonmembers)),
                    membership_checks='passed; both audit classes excluded from shadow graph and all attack splits',
                    pretraining_exclusions='never-member interactions, validation/test histories and social edges excluded',
                    method_adaptation='never-members excluded from retained candidate pools; requested histories contain only deleted original members',
                    selection='native retained validation; frozen attacker never tunes target model or method',
                    reuse='only original config and frozen independent shadow attacker reused, no target weights',
                    evaluation='fixed original membership labels; deleted members versus genuine never-members')
    atomic_json(base/'manifest.json', protocol)
    print('CONTROLLED_TARGET_MANIFEST', base/'manifest.json', flush=True)
    if args.prepare_only:
        return
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    if not args.evaluate_only:
        for method in args.methods:
            out = base/method; out.mkdir(exist_ok=True)
            prov = dict(provenance, method=method)
            with (out/'run.lock').open('w') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                if (out/'completion.json').exists():
                    if json.loads((out/'completion.json').read_text())['provenance'] != prov:
                        raise ValueError('Completed method provenance mismatch')
                    continue
                if method != 'original' and not (base/'original/completion.json').exists():
                    raise RuntimeError('Train the controlled original first; utility target weights are invalid here')
                seed_everything(args.seed); method_cfg = dict(cfg)
                synchronize(device)
                if device.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats(device)
                start = time.perf_counter()
                if method in ('original', 'retrain'):
                    data = before if method == 'original' else after
                    model = ODSR(data, method_cfg, device).to(device)
                    training = native_train(model, data, method_cfg, out, prov,
                                            cfg['original_epochs'], device)
                elif method in ('reset_graph_only', 'reset_finetune'):
                    source_method = 'graph_only' if method == 'reset_graph_only' else 'retain_finetune'
                    source_out = base/source_method
                    if not (source_out/'completion.json').exists():
                        raise RuntimeError(f'Run {source_method} before its reset control')
                    model, source_ck = load_model(source_out, after, raw_members, deleted, device)
                    if source_ck['provenance'] != dict(provenance, method=source_method):
                        raise ValueError('Reset source provenance mismatch')
                    training = dict(source_method=source_method, source_training=source_ck['training'],
                                    reset=reset_isolated_odsr_rows(model, after, deleted, args.seed + 510000),
                                    timing_scope='reset postprocessing only; add source method cost')
                else:
                    original, _ = load_model(base/'original', before, raw_members, deleted, device)
                    model = compact(original, after, method_cfg, device)
                    if method == 'retain_finetune':
                        training = native_train(model, after, method_cfg, out, prov,
                                                cfg['unlearn_epochs'], device)
                    else:
                        model, training = optimize(original, model, before, after, raw_members,
                                                   method_cfg, out, prov, device,
                                                   SimpleNamespace(method=method, seed=args.seed))
                    del original
                synchronize(device)
                elapsed = time.perf_counter() - start
                atomic_torch(out/'deployment.pt', dict(model=model.state_dict(), config=method_cfg,
                             provenance=prov, training=training, retained_original_ids=np.arange(raw.U),
                             requested_user_ids=deleted, never_member_user_ids=nonmembers))
                efficiency = dict(training_wall_seconds=elapsed,
                                  peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
                                  timing_mode='shared_GPU_unless_externally_isolated; graph_build_excluded')
                atomic_json(out/'completion.json', dict(provenance=prov, training=training, efficiency=efficiency))
                print('CONTROLLED_METHOD_COMPLETE', method, out, flush=True)
                del model
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            # Report each finished method promptly; attack outputs never enter training.
            evaluate_completed(base, result_dir, raw, raw_members, before, after, deleted,
                               nonmembers, bank, provenance, device, args.bootstrap_samples)
    else:
        evaluate_completed(base, result_dir, raw, raw_members, before, after, deleted,
                           nonmembers, bank, provenance, device, args.bootstrap_samples)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--fraction', type=float, default=.01)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--methods', nargs='+', choices=METHODS,
                        default=['original', 'retrain', 'graph_only', 'retain_finetune', 'ours',
                                 'reset_graph_only', 'reset_finetune'])
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--evaluate-only', action='store_true')
    parser.add_argument('--bootstrap-samples', type=int, default=1000)
    run(parser.parse_args())
