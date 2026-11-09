"""Train only the new native-topology candidate on an existing controlled target.

Reuse that target's completed original and scratch models, never utility-target
weights. Evaluate original, scratch and candidate with the same independently
fitted default and separately shadow-validation-tuned benchmark-feature attacks.
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

from run_unlearning import ROOT, ODSR, atomic_json, compact, digest, evaluate, load_data, seed_everything
from run_egcl_backbone import atomic_torch, synchronize
from run_controlled_target_mio import make_graphs, load_model
from run_odsr_native_topology import prepare, train_retained
from audit_corrected_forgetting import compare, probes, score_probe
from social_unlearning.benchmark_mio import BenchmarkMIO, embedding_features
from social_unlearning.controlled_mio import attack_metrics, bootstrap_attack
from social_unlearning.edge_deletion import filtered_raw


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def source_checkpoints(target, protocol):
    result = {}
    for method in ['original', 'retrain']:
        out = target/method
        if not (out/'completion.json').exists() or not (out/'deployment.pt').exists():
            raise RuntimeError(f'Controlled {method} must complete first: {out}')
        completion = json.loads((out/'completion.json').read_text())
        expected = dict(protocol['provenance'], method=method)
        if completion['provenance'] != expected:
            raise ValueError(f'Controlled {method} completion provenance mismatch')
        ck = torch.load(out/'deployment.pt', map_location='cpu', weights_only=False)
        if ck['provenance'] != expected:
            raise ValueError(f'Controlled {method} deployment provenance mismatch')
        result[method] = ck
    return result


def load_attacks(bank, tuned_bank, protocol, queries):
    manifest = json.loads((bank/'manifest.json').read_text())
    manifest_hash = sha(bank/'manifest.json')
    if manifest_hash != protocol['provenance']['shadow_manifest'] or manifest['provenance']['backbone'] != 'ODSR':
        raise ValueError('Wrong independent shadow bank for this controlled target')
    forbidden = np.concatenate([np.asarray(ids, dtype=np.int64) for ids in manifest['roles'].values()]
                               + [np.asarray(manifest['shadow_member_users'], dtype=np.int64)])
    if np.intersect1d(queries, forbidden).size or not np.isin(queries, manifest['excluded_target_users']).all():
        raise ValueError('Controlled audit identity overlap with shadow/attacker data')
    default = torch.load(bank/'attacker.pt', map_location='cpu', weights_only=False)
    if default['provenance'] != manifest['provenance']:
        raise ValueError('Default attacker/manifest provenance mismatch')
    if sha(bank/'attacker.pt') != protocol['provenance']['attacker']:
        raise ValueError('Default attacker changed since controlled target provisioning')
    tuned = torch.load(tuned_bank/'attacker.pt', map_location='cpu', weights_only=False)
    validation = json.loads((tuned_bank/'validation.json').read_text())
    if tuned['original_shadow_manifest'] != manifest_hash or validation['source'] != tuned['source']:
        raise ValueError('Tuned attacker/validation/shadow provenance mismatch')
    if not torch.isfinite(tuned['mean']).all() or not torch.isfinite(tuned['std']).all() or (tuned['std'] <= 0).any():
        raise ValueError('Invalid shadow-training normalization')
    if not np.isfinite(tuned['threshold']):
        raise ValueError('Invalid tuned shadow-validation threshold')
    return dict(default=dict(checkpoint=default, mean=None, std=None, threshold=.5,
                             path=str(bank/'attacker.pt'), sha256=sha(bank/'attacker.pt'),
                             selection='benchmark-default frozen attacker; fixed threshold .5'),
                tuned=dict(checkpoint=tuned, mean=tuned['mean'], std=tuned['std'], threshold=float(tuned['threshold']),
                           path=str(tuned_bank/'attacker.pt'), sha256=sha(tuned_bank/'attacker.pt'),
                           validation_path=str(tuned_bank/'validation.json'), validation_sha256=sha(tuned_bank/'validation.json'),
                           selection=validation['selection'], shadow_test=validation['shadow_test']))


def evaluate_models(models, raw, after, deleted, nonmembers, attacks, bootstrap_samples=1000):
    histories = defaultdict(list)
    for user, item in np.concatenate([raw.train, raw.val]):
        histories[int(user)].append(int(item))
    queries = np.concatenate([deleted, nonmembers])
    labels = np.concatenate([np.ones(len(deleted), dtype=np.int64), np.zeros(len(nonmembers), dtype=np.int64)])
    deletion_probe = probes(raw, deleted, raw.meta['seed'])
    outputs = {}; records = {}; attack_rows = {name: [] for name in attacks}
    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            users, items = model()
            x = embedding_features(users.cpu(), items.cpu(), queries,
                                   {user: user for user in range(raw.U)}, histories)
        utility, _ = evaluate(model, after, after.test, model.device)
        records[name] = score_probe((users.cpu(), items.cpu()), deletion_probe)
        outputs[name] = dict(retained_utility=utility,
                             direct_forgetting={key: float(np.mean([row[key] for row in records[name]]))
                                                for key in ['positive_beats_negative', 'mean_pair_margin', 'mean_pair_bpr_loss']})
        for attack_name, saved in attacks.items():
            attacker = BenchmarkMIO(x.shape[1]); attacker.load_state_dict(saved['checkpoint']['model'])
            features = x if saved['mean'] is None else (x - saved['mean']) / saved['std']
            probability = attacker.member_probability(features).numpy()
            attack_rows[attack_name].append(dict(method=name,
                metrics=attack_metrics(labels, probability, saved['threshold']),
                uncertainty=bootstrap_attack(labels, probability, saved['threshold'], raw.meta['seed'], bootstrap_samples),
                probabilities=probability.tolist()))
        del users, items, x
    for name in outputs:
        outputs[name]['direct_forgetting'].update(compare(records[name], records['retrain']))
    return dict(models=outputs, attacks={name: dict(
                    **{key: value for key, value in saved.items() if key not in ['checkpoint', 'mean', 'std']},
                    rows=attack_rows[name]) for name, saved in attacks.items()},
                query_users=queries.tolist(), original_membership_labels=labels.tolist(),
                interpretation='Compare original discrimination and FPR first. Default and tuned attacks remain separate. Mean probabilities are not necessarily target-calibrated. Chance AUC is not a guarantee, and this single-seed bootstrap does not establish method superiority.')


def run(args):
    torch.set_num_threads(args.cpu_threads); device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    target = args.target.resolve(); protocol = json.loads((target/'manifest.json').read_text())
    source = source_checkpoints(target, protocol)
    seed = source['original']['config']['seed']
    raw = load_data(args.dataset, seed)
    if raw.meta['split_hash'] != protocol['provenance']['split']:
        raise ValueError('Controlled source dataset/split mismatch')
    deleted = np.asarray(protocol['requested_users'], dtype=np.int64)
    nonmembers = np.asarray(protocol['never_member_users'], dtype=np.int64)
    queries = np.concatenate([deleted, nonmembers])
    if len(np.unique(queries)) != len(queries) or not len(deleted) or not len(nonmembers):
        raise ValueError('Both controlled membership groups must be nonempty and disjoint')
    bank = Path(protocol['frozen_attacker']).parent if args.bank is None else args.bank.resolve()
    attacks = load_attacks(bank, args.tuned_bank.resolve(), protocol, queries)
    shadow_raw = load_data(args.dataset, 42)
    manifest = json.loads((bank/'manifest.json').read_text())
    if manifest['provenance']['split'] != shadow_raw.meta['split_hash'] or raw.um != shadow_raw.um or raw.im != shadow_raw.im:
        raise ValueError('Shadow/target mappings or data differ')
    cfg = dict(source['original']['config'], lr=.0001, epochs=5, weight_mix=.25,
               correction_relative_step=.001, probe_size=1024, variant='full',
               method='controlled_native_topology_full', checkpoint_selection='retained_validation_recall@20')
    files = [Path(__file__), ROOT/'scripts/run_odsr_native_topology.py', ROOT/'scripts/run_controlled_target_mio.py',
             ROOT/'scripts/run_unlearning.py', ROOT/'scripts/run_egcl_backbone.py', ROOT/'scripts/audit_corrected_forgetting.py',
             ROOT/'external/ODSR/ODSR.py']
    files += [ROOT/'src/social_unlearning'/name for name in
              ['odsr_native_topology.py', 'isolated_reset.py', 'edge_deletion.py', 'native_guarded.py',
               'native_guarded_memory.py', 'controlled_mio.py', 'benchmark_mio.py', 'checkpoint.py']]
    hashes = {str(path.relative_to(ROOT)): sha(path) for path in files}
    prov = dict(protocol='controlled_odsr_native_topology_v1',
                source=hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
                target_manifest=sha(target/'manifest.json'), original_checkpoint=sha(target/'original/deployment.pt'),
                scratch_checkpoint=sha(target/'retrain/deployment.pt'), split=raw.meta['split_hash'],
                deleted=digest(deleted), never_members=digest(nonmembers),
                config=hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
                deletion_semantics='incident_edges_only_keep_user_rows')
    version = hashlib.sha256(json.dumps(prov, sort_keys=True).encode()).hexdigest()[:16]
    suffix = Path('ODSR')/args.dataset/f'seed_{seed}'/target.name/version
    out = ROOT/'checkpoints/controlled_odsr_native_topology_v1'/suffix
    result_dir = ROOT/'results/controlled_odsr_native_topology_v1'/suffix
    out.mkdir(parents=True, exist_ok=True)
    with (out/'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        before, after = make_graphs(raw, deleted, nonmembers, device)
        raw_members = filtered_raw(raw, nonmembers)
        seed_everything(seed)
        original = ODSR(before, cfg, device).to(device); original.load_state_dict(source['original']['model'])
        candidate = compact(original, after, cfg, device)
        if not (out/'completion.json').exists():
            synchronize(device)
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            resumed = (out/'last.pt').exists(); start = time.perf_counter()
            preparation_path = out/'preparation.pt'
            if preparation_path.exists():
                prepared = torch.load(preparation_path, map_location=device, weights_only=False)
                if prepared['provenance'] != prov:
                    raise ValueError('Controlled candidate preparation provenance mismatch')
                candidate.load_state_dict(prepared['model']); weights = prepared['weights']; metadata = prepared['metadata']
            else:
                weights, metadata = prepare(original, candidate, before, after, raw_members, cfg, seed, 'full')
                atomic_torch(preparation_path, dict(model=candidate.state_dict(), weights=weights,
                                                   metadata=metadata, provenance=prov))
            atomic_json(out/'manifest.json', dict(provenance=prov, config=cfg, source_files=hashes,
                        controlled_target=str(target), requested_users=deleted.tolist(), never_member_users=nonmembers.tolist(),
                        preparation=metadata, reused='completed controlled original/scratch only; no recommender pretraining',
                        selection='fixed preset; retained validation only; attackers do not enter training'))
            training = train_retained(candidate, after, weights, cfg, prov, out, device)
            synchronize(device); elapsed = time.perf_counter() - start
            efficiency = dict(candidate_wall_seconds=elapsed, resumed=resumed,
                              peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
                              timing_mode='shared_GPU_not_isolated; graph/model reconstruction excluded; earlier costs excluded after resume')
            atomic_torch(out/'deployment.pt', dict(model=candidate.state_dict(), config=cfg, provenance=prov,
                         training=training, retained_original_ids=np.arange(raw.U), requested_user_ids=deleted,
                         never_member_user_ids=nonmembers))
            atomic_json(out/'completion.json', dict(provenance=prov, training=training, efficiency=efficiency))
        else:
            completion = json.loads((out/'completion.json').read_text())
            saved = torch.load(out/'deployment.pt', map_location='cpu', weights_only=False)
            if completion['provenance'] != prov or saved['provenance'] != prov:
                raise ValueError('Completed controlled candidate provenance mismatch')
            candidate.load_state_dict(saved['model'])
        reference = ODSR(after, source['retrain']['config'], device).to(device)
        reference.load_state_dict(source['retrain']['model'])
        report = evaluate_models(dict(original=original, retrain=reference, native_topology_full=candidate),
                                 raw, after, deleted, nonmembers, attacks, args.bootstrap_samples)
        completion = json.loads((out/'completion.json').read_text())
        report.update(status='completed_controlled_native_topology_audit', backbone='ODSR', dataset=args.dataset,
                      seed=seed, method='native_topology_full', provenance=prov, source_target_protocol=protocol['provenance'],
                      config=cfg, training=completion['training'], efficiency=completion['efficiency'],
                      checkpoint=str(out/'deployment.pt'),
                      feature_schema='concat(native propagated user, mean native item of original train+validation known history)',
                      limitation='Finite-probe first-order correction and isolated reset, not a certificate. Direct ranks and MIO are complementary; no win is presumed.')
        attack_version = hashlib.sha256(json.dumps({name: saved['sha256'] for name, saved in attacks.items()}, sort_keys=True).encode()).hexdigest()[:16]
        atomic_json(result_dir/attack_version/'results.json', report)
        print('CONTROLLED_NATIVE_TOPOLOGY_RESULT', result_dir/attack_version/'results.json', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--target', required=True, type=Path)
    parser.add_argument('--tuned-bank', required=True, type=Path)
    parser.add_argument('--bank', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--bootstrap-samples', type=int, default=1000)
    args = parser.parse_args()
    if args.bootstrap_samples < 1:
        parser.error('Positive bootstrap count required')
    run(args)
