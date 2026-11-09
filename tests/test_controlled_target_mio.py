import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'src')]
from test_edge_deletion import fixture
from run_unlearning import ODSR
from run_controlled_target_mio import evaluate_completed, make_graphs, native_train
from social_unlearning.benchmark_mio import BenchmarkMIO
from social_unlearning.controlled_mio import attack_metrics, target_partition
from social_unlearning.edge_deletion import filtered_raw, retained_proxy
from social_unlearning.baselines import RequestGraph


def manifest():
    return dict(excluded_target_users=[0, 1, 2, 3, 4, 5], shadow_member_users=[6, 8],
                roles=dict(train=[6, 7], validation=[8, 9], test=[10, 11]))


def test_partition_is_disjoint_reproducible_and_refuses_leakage():
    a, n = target_partition(12, manifest(), 42, .2)
    aa, nn = target_partition(12, manifest(), 42, .2)
    np.testing.assert_array_equal(a, aa); np.testing.assert_array_equal(n, nn)
    assert len(a) == len(n) == 2 and not np.intersect1d(a, n).size
    assert set(np.concatenate([a, n])).issubset(set(range(6)))
    bad = manifest(); bad['roles']['test'].append(0)
    with pytest.raises(ValueError, match='overlaps'):
        target_partition(12, bad, 42, .2)
    with pytest.raises(ValueError, match='too small'):
        target_partition(12, manifest(), 42, .4)


def test_never_members_absent_before_training_and_not_request_histories():
    raw, cfg = fixture(); before, after = make_graphs(raw, [1], [3], 'cpu')
    assert 1 in before.train[:, 0] and 3 not in before.train[:, 0]
    assert 1 not in after.train[:, 0] and 3 not in after.train[:, 0]
    assert not after.adj_uu.to_dense()[3].any()  # no synthetic absent-user loop
    assert set(after.retained_users) == {0, 2, 4}
    original = ODSR(before, cfg, 'cpu'); student = ODSR(after, cfg, 'cpu')
    request = RequestGraph(original, student, retained_proxy(after), filtered_raw(raw, [3]), 42)
    assert set(request.forget_pairs[:, 0]) == {1}
    assert original()[0].shape == student()[0].shape == (5, 4)


def test_metrics_use_both_original_membership_classes():
    result = attack_metrics([1, 1, 0, 0], [.9, .8, .2, .1])
    assert result['auc'] == result['balanced_accuracy'] == 1
    assert result['members'] == result['nonmembers'] == 2
    with pytest.raises(ValueError, match='both actual membership'):
        attack_metrics([1, 1], [.9, .8])


def test_native_training_resume_matches_uninterrupted(tmp_path):
    raw, cfg = fixture(); before, _ = make_graphs(raw, [1], [3], 'cpu')
    cfg.update(seed=42, lr=.001, batch_size=4, patience=5)
    torch.manual_seed(3); initial = ODSR(before, cfg, 'cpu').state_dict()
    provenance = dict(test='controlled-native-resume')
    a = tmp_path/'a'; b = tmp_path/'b'; a.mkdir(); b.mkdir()
    full = ODSR(before, cfg, 'cpu'); full.load_state_dict(initial)
    native_train(full, before, cfg, a, provenance, 2, torch.device('cpu'))
    partial = ODSR(before, cfg, 'cpu'); partial.load_state_dict(initial)
    native_train(partial, before, cfg, b, provenance, 1, torch.device('cpu'))
    # Simulate interruption after epoch one, before a configured two-epoch budget.
    ck = torch.load(b/'last.pt', weights_only=False)
    ck['metrics']['progress']['complete'] = False
    torch.save(ck, b/'last.pt')
    resumed = ODSR(before, cfg, 'cpu')
    native_train(resumed, before, cfg, b, provenance, 2, torch.device('cpu'))
    for key, value in full.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[key], rtol=0, atol=0)


def test_completed_targets_have_real_two_class_report(tmp_path):
    raw, cfg = fixture(); raw.meta = dict(dataset='toy', seed=42)
    deleted = np.array([1]); absent = np.array([3])
    before, after = make_graphs(raw, deleted, absent, 'cpu')
    base = tmp_path/'targets'; bank = tmp_path/'bank'; bank.mkdir()
    torch.save(dict(model=BenchmarkMIO(8).state_dict()), bank/'attacker.pt')
    provenance = dict(test='controlled-report')
    for method, data in [('original', before), ('retrain', after)]:
        out = base/method; out.mkdir(parents=True)
        prov = dict(provenance, method=method)
        torch.save(dict(model=ODSR(data, cfg, 'cpu').state_dict(), config=cfg, provenance=prov),
                   out/'deployment.pt')
        (out/'completion.json').write_text(json.dumps(dict(provenance=prov, training={}, efficiency={})))
    evaluate_completed(base, tmp_path/'results', raw, filtered_raw(raw, absent), before, after,
                       deleted, absent, bank, provenance, torch.device('cpu'), 5)
    report = json.loads((tmp_path/'results/results.json').read_text())
    assert report['original_membership_labels'] == [1, 0]
    assert report['query_users'] == [1, 3]
    assert len(report['rows']) == 2
    for row in report['rows']:
        assert row['metrics_at_0_5']['members'] == row['metrics_at_0_5']['nonmembers'] == 1
        assert 'paired_mean_probability_gap_to_retrain' in row
