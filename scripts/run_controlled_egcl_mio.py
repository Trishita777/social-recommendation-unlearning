"""Independent EGCL target suite with genuine never-history-members and frozen MIO.

Full ID rows, native full contrastive candidate pools and global regularization
are preserved. A nonmember contributes no interaction/social history, although
its empty row remains in these native objectives. All target audit identities
must be absent from the shadow graph and every attacker fitting/evaluation split.
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

from run_unlearning import ROOT, atomic_json, digest, evaluate, load_data, seed_everything
from run_egcl_backbone import EGCL, atomic_torch, load_parameters, parameter_state, synchronize, train
from run_egcl_unlearning_adapters import train_gnndelete
from social_unlearning.benchmark_mio import BenchmarkMIO, embedding_features
from social_unlearning.controlled_mio import (attack_metrics, bootstrap_attack,
                                              target_partition, validate_graph_membership)
from social_unlearning.edge_deletion import edge_graph, filtered_raw
from social_unlearning.egcl_unlearning_adapters import (GNNDeleteEGCL, deletion_setup,
                                                       topology_correction)

METHODS = ['original', 'retrain', 'graph_only', 'retain_finetune', 'native_topology', 'gnndelete_native']


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def make_graphs(raw, deleted, nonmembers, device):
    before = edge_graph(raw, nonmembers, device)
    after = edge_graph(raw, np.union1d(deleted, nonmembers), device)
    before.deleted_users = np.array([], dtype=np.int64)
    before.retained_users = np.setdiff1d(np.arange(raw.U), nonmembers)
    after.deleted_users = np.asarray(deleted)
    after.retained_users = np.setdiff1d(np.arange(raw.U), np.union1d(deleted, nonmembers))
    validate_graph_membership(before, deleted, nonmembers, before_deletion=True)
    validate_graph_membership(after, deleted, nonmembers, before_deletion=False)
    for data in [before, after]:
        del data.adj_ui, data.adj_uu, data.norm_adj
    return before, after


def load_model(out, data, raw_members, deleted, device):
    ck = torch.load(out/'deployment.pt', map_location='cpu', weights_only=False)
    model = EGCL(data, ck['config'], device)
    if out.name == 'gnndelete_native':
        masks, _, _, _ = deletion_setup(raw_members, deleted, model.layers,
                                        ck['config']['seed'], ck['config']['request_edge_cap'])
        model = GNNDeleteEGCL(model, masks, ROOT/'external/baseline_sources/GNNDelete')
    load_parameters(model, ck['model']); model.eval()
    return model, ck


def fit_method(method, base, raw, before, after, raw_members, deleted, cfg, provenance, device):
    out = base/method; out.mkdir(exist_ok=True)
    prov = dict(provenance, method=method)
    if (out/'completion.json').exists():
        if json.loads((out/'completion.json').read_text())['provenance'] != prov:
            raise ValueError('Controlled EGCL completion provenance mismatch')
        return
    if method != 'original' and not (base/'original/completion.json').exists():
        raise RuntimeError('Controlled original must finish first; utility originals are invalid target weights')
    method_cfg = dict(cfg, method=method)
    if method in ('retain_finetune', 'native_topology'):
        method_cfg.update(epochs=20, min_epochs=20)
    seed_everything(cfg['seed']); synchronize(device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    resumed = (out/'last.pt').exists(); start = time.perf_counter()
    data = before if method == 'original' else after
    model = EGCL(data, method_cfg, device)
    if method in ('original', 'retrain'):
        training = train(model, data, method_cfg, prov, out, device)
    else:
        original, original_ck = load_model(base/'original', before, raw_members, deleted, device)
        if original_ck['provenance'] != dict(provenance, method='original'):
            raise ValueError('Controlled original provenance mismatch')
        load_parameters(model, original_ck['model'])
        if method == 'graph_only':
            training = dict(epoch=0, optimizer_seconds=0., complete=True)
        elif method == 'retain_finetune':
            training = train(model, data, method_cfg, prov, out, device)
        elif method == 'native_topology':
            correction_path = out/'correction.pt'
            if correction_path.exists():
                saved = torch.load(correction_path, map_location='cpu', weights_only=False)
                if saved['provenance'] != prov:
                    raise ValueError('Controlled topology correction provenance mismatch')
                load_parameters(model, saved['model']); correction = saved['metadata']
            else:
                correction = topology_correction(original, model, deleted, cfg['correction_relative_step'])
                atomic_torch(correction_path, dict(model=parameter_state(model), provenance=prov, metadata=correction))
            training = train(model, data, method_cfg, prov, out, device)
            training['topology_correction'] = correction
        elif method == 'gnndelete_native':
            model, training = train_gnndelete(original, model, raw_members, data, method_cfg, prov,
                                              out, ROOT/'external/baseline_sources/GNNDelete',
                                              device, cfg['seed'])
        else:
            raise ValueError('Unsupported controlled EGCL method')
        del original
    synchronize(device); elapsed = time.perf_counter() - start
    efficiency = dict(training_wall_seconds=elapsed, resumed=resumed,
                      wall_scope='current invocation; graph construction excluded; earlier costs excluded after resume',
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
                      timing_mode='shared_GPU_unless_externally_isolated')
    atomic_torch(out/'deployment.pt', dict(model=parameter_state(model), config=method_cfg, provenance=prov,
                 retained_original_ids=np.arange(raw.U), requested_user_ids=deleted,
                 training=training, graph_buffers='rebuild_from_verified_data'))
    atomic_json(out/'completion.json', dict(provenance=prov, training=training, efficiency=efficiency))
    print('CONTROLLED_EGCL_METHOD_COMPLETE', method, out, flush=True)
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()


def evaluate_completed(base, result_dir, raw, raw_members, before, after, deleted,
                       nonmembers, bank, provenance, device, bootstrap_samples=1000):
    saved = torch.load(bank/'attacker.pt', map_location='cpu', weights_only=False)
    histories = defaultdict(list)
    for user, item in np.concatenate([raw.train, raw.val]):
        histories[int(user)].append(int(item))
    queries = np.concatenate([deleted, nonmembers])
    labels = np.concatenate([np.ones(len(deleted), dtype=np.int64), np.zeros(len(nonmembers), dtype=np.int64)])
    threshold = .5; threshold_source = 'fixed_0.5; no existing shadow-validation threshold file'
    if (bank/'diagnostics.json').exists():
        diagnostic = json.loads((bank/'diagnostics.json').read_text())
        if 'validation_selected_threshold' in diagnostic:
            threshold = float(diagnostic['validation_selected_threshold'])
            threshold_source = dict(path=str(bank/'diagnostics.json'), sha256=sha(bank/'diagnostics.json'),
                                    selection='existing shadow validation only')
    rows = []; probabilities = {}
    for method in METHODS:
        out = base/method
        if not (out/'completion.json').exists():
            continue
        completion = json.loads((out/'completion.json').read_text())
        if completion['provenance'] != dict(provenance, method=method):
            raise ValueError('Controlled EGCL result provenance mismatch')
        model, ck = load_model(out, before if method == 'original' else after,
                               raw_members, deleted, device)
        if ck['provenance'] != completion['provenance']:
            raise ValueError('Controlled EGCL deployment provenance mismatch')
        with torch.no_grad():
            users, items = model()
            x = embedding_features(users.cpu(), items.cpu(), queries,
                                   {user: user for user in range(raw.U)}, histories)
            norm = users[torch.as_tensor(queries, device=device)].norm(dim=1).cpu().numpy()
        if not np.all(norm[len(deleted):] == 0) or (method != 'original' and not np.all(norm == 0)):
            raise ValueError('EGCL isolated audit users must have zero native representations')
        attacker = BenchmarkMIO(x.shape[1]); attacker.load_state_dict(saved['model'])
        scores = attacker.member_probability(x).numpy(); probabilities[method] = scores
        utility, _ = evaluate(model, after, after.test, device)
        rows.append(dict(method=method, metrics_at_0_5=attack_metrics(labels, scores),
                         metrics_at_shadow_validation_threshold=attack_metrics(labels, scores, threshold),
                         uncertainty=bootstrap_attack(labels, scores, threshold, raw.meta['seed'], bootstrap_samples),
                         mean_native_user_norm=dict(deleted_members=float(norm[:len(deleted)].mean()),
                                                    never_members=float(norm[len(deleted):].mean())),
                         retained_utility=utility, training=completion['training'], efficiency=completion['efficiency']))
        del model, users, items, x, attacker
    if 'retrain' in probabilities:
        for row in rows:
            gap = probabilities[row['method']] - probabilities['retrain']
            row['paired_mean_probability_gap_to_retrain'] = dict(deleted_members=float(gap[:len(deleted)].mean()),
                                                                never_members=float(gap[len(deleted):].mean()))
    report = dict(status='controlled_EGCL_target_audit', backbone='EGCL', dataset=raw.meta['dataset'],
                  seed=raw.meta['seed'], provenance=provenance, rows=rows, threshold_source=threshold_source,
                  query_users=queries.tolist(), original_membership_labels=labels.tolist(),
                  probabilities={method: scores.tolist() for method, scores in probabilities.items()},
                  pending_methods=[method for method in METHODS if method not in probabilities],
                  unavailable_methods=dict(scif='Native alternating EGCL stationarity requires a coupled influence derivation; existing symmetric user-table solver is not faithful',
                                           da2='Requested native predictions have zero forgetting gradient; no validated native objective/schedule adaptation'),
                  features='concat(native user embedding, mean native item embedding of known original train+validation history)',
                  threat_model='native representation access and known history; frozen independent shadow attacker',
                  native_objective='full IDs, all native CL candidates/anchors and full-table L2 retained; never-members contribute no history edges',
                  interpretation='Separate controlled targets, not existing utility targets. Original-membership labels remain fixed after deletion. Validate original target attack strength first. All post-deletion user feature halves are structurally zero; chance attack scores or zero native predictions alone do not prove shared-influence removal. Whole-user bootstrap conditions on one target and one attacker.')
    atomic_json(result_dir/'results.json', report)
    print('CONTROLLED_EGCL_MIO_RESULT', result_dir/'results.json', flush=True)


def run(args):
    torch.set_num_threads(args.cpu_threads)
    device = torch.device('cpu' if args.prepare_only else args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    bank = args.bank.resolve()
    manifest = json.loads((bank/'manifest.json').read_text())
    if manifest['provenance']['backbone'] != 'EGCL' or manifest['config']['seed'] != 1042:
        raise ValueError('Expected the independent EGCL shadow seed1042 bank')
    attacker = torch.load(bank/'attacker.pt', map_location='cpu', weights_only=False)
    if attacker['provenance'] != manifest['provenance']:
        raise ValueError('Frozen EGCL attacker provenance differs from its shadow manifest')
    del attacker
    raw = load_data(args.dataset, args.seed); shadow_raw = load_data(args.dataset, 42)
    if manifest['provenance']['split'] != shadow_raw.meta['split_hash']:
        raise ValueError('Shadow bank dataset/split mismatch')
    if raw.um != shadow_raw.um or raw.im != shadow_raw.im:
        raise ValueError('Target/shadow user or item mappings differ')
    deleted, nonmembers = target_partition(raw.U, manifest, args.seed, args.fraction)
    path = ROOT/'checkpoints/EGCL'/args.dataset/f'seed_{args.seed}'/raw.meta['split_hash'][:16]/'original/deployment.pt'
    ck = torch.load(path, map_location='cpu', weights_only=False)
    cfg = dict(ck['config'], seed=args.seed, controlled_target_mio=True,
               deletion_semantics='incident_edges_only_keep_user_rows',
               correction_relative_step=args.relative_step, deletion_steps=args.steps,
               deletion_lr=.001, deletion_validation_every=25, deletion_patience=10,
               request_edge_cap=4096, alpha=.5)
    files = [Path(__file__), ROOT/'scripts/run_egcl_backbone.py', ROOT/'scripts/run_unlearning.py',
             ROOT/'scripts/run_egcl_unlearning_adapters.py']
    files += [ROOT/'src/social_unlearning'/name for name in
              ['egcl.py', 'egcl_unlearning_adapters.py', 'gnndelete_edges.py', 'edge_deletion.py',
               'controlled_mio.py', 'benchmark_mio.py']]
    files += [ROOT/'external/baseline_sources/GNNDelete'/name for name in
              ['framework/models/deletion.py', 'framework/trainer/gnndelete_nodeemb.py']]
    file_hashes = {str(p.relative_to(ROOT)): sha(p) for p in files}
    provenance = dict(protocol='controlled_EGCL_target_mio_v1', split=raw.meta['split_hash'],
                      source=hashlib.sha256(json.dumps(file_hashes, sort_keys=True).encode()).hexdigest(),
                      data=raw.meta['source_files'], social=raw.meta['social_hash'],
                      shadow_manifest=sha(bank/'manifest.json'), attacker=sha(bank/'attacker.pt'),
                      deleted=digest(deleted), never_members=digest(nonmembers),
                      config=hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
                      deletion_semantics=cfg['deletion_semantics'])
    experiment = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()[:16]
    suffix = Path('EGCL')/args.dataset/f'seed_{args.seed}'/experiment
    base = ROOT/'checkpoints/controlled_egcl_target_mio'/suffix
    result_dir = ROOT/'results/controlled_egcl_target_mio'/suffix
    base.mkdir(parents=True, exist_ok=True)
    with (base/'suite.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        before, after = make_graphs(raw, deleted, nonmembers, device)
        raw_members = filtered_raw(raw, nonmembers)
        protocol = dict(provenance=provenance, source_files=file_hashes, config=cfg,
                        frozen_attacker=str(bank/'attacker.pt'), requested_users=deleted.tolist(),
                        never_member_users=nonmembers.tolist(), full_users=raw.U,
                        requested_fraction_of_full_universe=args.fraction,
                        requested_fraction_of_original_members=len(deleted)/(raw.U-len(nonmembers)),
                        checks='disjoint shadow/target audit identities; genuine original members; no never-member incident edges; full ID mapping',
                        selection='native retained validation recall@20; attack outputs never tune target training',
                        reused='utility original config only; fresh target weights; frozen independent shadow attacker',
                        native_objective='full contrastive candidates/anchors and full-table regularization including empty rows, matching native fresh reference',
                        native_topology='bounded first-order native gradient-difference correction then native alternating training; no certificate',
                        gnndelete_native='released DeletionLayer/nodeemb MSE form adapted to shared operators across EGCL streams; original-graph witnesses, altered-graph locality, typed negatives',
                        adaptation_status='SCIF and DA2 unavailable; native_topology and gnndelete_native are disclosed adaptations')
        atomic_json(base/'manifest.json', protocol)
        print('CONTROLLED_EGCL_TARGET_MANIFEST', base/'manifest.json', flush=True)
        if args.prepare_only:
            return
        if args.evaluate_only:
            evaluate_completed(base, result_dir, raw, raw_members, before, after, deleted,
                               nonmembers, bank, provenance, device, args.bootstrap_samples)
        else:
            for method in args.methods:
                fit_method(method, base, raw, before, after, raw_members, deleted, cfg, provenance, device)
                evaluate_completed(base, result_dir, raw, raw_members, before, after, deleted,
                                   nonmembers, bank, provenance, device, args.bootstrap_samples)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--bank', required=True, type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--fraction', type=float, default=.01)
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=METHODS)
    parser.add_argument('--steps', type=int, default=1500)
    parser.add_argument('--relative-step', type=float, default=.001)
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--bootstrap-samples', type=int, default=1000)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--evaluate-only', action='store_true')
    args = parser.parse_args()
    if args.steps < 1 or args.bootstrap_samples < 1:
        parser.error('Positive training/bootstrap budgets required')
    run(args)
