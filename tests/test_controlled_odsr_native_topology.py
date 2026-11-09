import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'src')]
from test_edge_deletion import fixture
from run_controlled_target_mio import make_graphs
from run_controlled_odsr_native_topology import evaluate_models, load_attacks, source_checkpoints
from run_odsr_native_topology import prepare, train_retained
from run_unlearning import ODSR, compact
from social_unlearning.benchmark_mio import BenchmarkMIO
from social_unlearning.edge_deletion import filtered_raw


def attack_banks(tmp_path):
    bank = tmp_path/'default'; tuned = tmp_path/'tuned'; bank.mkdir(); tuned.mkdir()
    manifest = dict(provenance=dict(backbone='ODSR', test='independent-shadow'),
                    roles=dict(train=[0], validation=[2], test=[4]),
                    shadow_member_users=[0, 2], excluded_target_users=[1, 3])
    (bank/'manifest.json').write_text(json.dumps(manifest))
    manifest_hash = hashlib.sha256((bank/'manifest.json').read_bytes()).hexdigest()
    torch.save(dict(model=BenchmarkMIO(8).state_dict(), provenance=manifest['provenance']), bank/'attacker.pt')
    torch.save(dict(model=BenchmarkMIO(8).state_dict(), mean=torch.zeros(8), std=torch.ones(8),
                    threshold=.4, source='tuned-source', original_shadow_manifest=manifest_hash), tuned/'attacker.pt')
    (tuned/'validation.json').write_text(json.dumps(dict(source='tuned-source',
                      selection='shadow validation only', shadow_test=dict(auc=.6))))
    protocol = dict(provenance=dict(shadow_manifest=manifest_hash,
                                    attacker=hashlib.sha256((bank/'attacker.pt').read_bytes()).hexdigest()))
    return bank, tuned, protocol


def test_two_attack_provenance_and_identity_guards(tmp_path):
    bank, tuned, protocol = attack_banks(tmp_path)
    attacks = load_attacks(bank, tuned, protocol, [1, 3])
    assert attacks['default']['threshold'] == .5 and attacks['tuned']['threshold'] == .4
    with pytest.raises(ValueError, match='identity overlap'):
        load_attacks(bank, tuned, protocol, [1, 2])
    changed = torch.load(tuned/'attacker.pt', weights_only=False)
    changed['original_shadow_manifest'] = 'wrong'
    torch.save(changed, tuned/'attacker.pt')
    with pytest.raises(ValueError, match='Tuned attacker'):
        load_attacks(bank, tuned, protocol, [1, 3])


def test_matching_controlled_original_and_scratch_required(tmp_path):
    protocol = dict(provenance={'controlled': 'same-experiment'})
    with pytest.raises(RuntimeError, match='original must complete'):
        source_checkpoints(tmp_path, protocol)
    for name in ['original', 'retrain']:
        out = tmp_path/name; out.mkdir()
        provenance = dict(protocol['provenance'], method=name)
        (out/'completion.json').write_text(json.dumps(dict(provenance=provenance)))
        torch.save(dict(provenance=provenance, model={}, config={}), out/'deployment.pt')
    assert set(source_checkpoints(tmp_path, protocol)) == {'original', 'retrain'}
    (tmp_path/'retrain/completion.json').write_text(json.dumps(dict(provenance={'wrong': True})))
    with pytest.raises(ValueError, match='retrain completion'):
        source_checkpoints(tmp_path, protocol)


def test_candidate_on_controlled_graph_and_both_attacks_report_all_controls(tmp_path):
    raw, cfg = fixture(); raw.meta = dict(seed=42, dataset='toy')
    cfg.update(seed=42, lr=.0001, epochs=2, batch_size=4, weight_mix=.25,
               probe_size=1024, correction_relative_step=.001)
    deleted = np.array([1]); absent = np.array([3])
    before, after = make_graphs(raw, deleted, absent, 'cpu')
    original = ODSR(before, cfg, 'cpu'); candidate = compact(original, after, cfg, 'cpu')
    teacher_state = {k: v.clone() for k, v in original.state_dict().items()}
    weights, metadata = prepare(original, candidate, before, after, filtered_raw(raw, absent), cfg, 42, 'full')
    assert metadata['correction']['original_triplets'] == len(before.train)
    assert len(before.train) == len(raw.train) - 2
    training_dir = tmp_path/'candidate'; training_dir.mkdir()
    train_retained(candidate, after, weights, cfg, {'test': 'controlled-candidate'}, training_dir, torch.device('cpu'))
    for key, value in original.state_dict().items():
        torch.testing.assert_close(value, teacher_state[key], rtol=0, atol=0)
    bank, tuned, protocol = attack_banks(tmp_path)
    attacks = load_attacks(bank, tuned, protocol, [1, 3])
    report = evaluate_models(dict(original=original, retrain=ODSR(after, cfg, 'cpu'),
                                  native_topology_full=candidate), raw, after, deleted, absent, attacks, 5)
    assert report['original_membership_labels'] == [1, 0]
    assert set(report['attacks']) == {'default', 'tuned'}
    assert len(report['models']) == 3
    for attack in report['attacks'].values():
        assert {row['method'] for row in attack['rows']} == {'original', 'retrain', 'native_topology_full'}
        assert all('fpr' in row['metrics'] and row['metrics']['nonmembers'] == 1 for row in attack['rows'])
    assert 'deleted_user_score_rmse' in report['models']['native_topology_full']['direct_forgetting']
