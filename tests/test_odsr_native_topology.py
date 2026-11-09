import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'src')]
from test_edge_deletion import fixture
from run_unlearning import ODSR, compact, graph
from run_odsr_native_topology import prepare, train_retained
from social_unlearning.edge_deletion import edge_graph
from social_unlearning.isolated_reset import reset_isolated_odsr_rows
from social_unlearning.odsr_native_topology import paired_probe, topology_correction


def setup():
    raw, cfg = fixture()
    cfg.update(seed=42, lr=.0001, epochs=2, batch_size=4, weight_mix=.25,
               correction_relative_step=.001, probe_size=1024)
    before = graph(raw, np.arange(raw.U), 'cpu'); after = edge_graph(raw, [1], 'cpu')
    original = ODSR(before, cfg, 'cpu'); student = compact(original, after, cfg, 'cpu')
    return raw, cfg, before, after, original, student


def test_paired_native_probe_and_bounded_correction_preserve_original_and_deleted_row():
    raw, cfg, before, after, original, student = setup()
    original_state = {k: v.clone() for k, v in original.state_dict().items()}
    deleted_row = student.embedding_dict['user_emb'][1].detach().clone()
    probe = paired_probe(before, [1], 991, 1024)
    assert len(probe['pairs']) == len(raw.train) and (~probe['retained']).sum() == 2
    assert all(n not in before.positives[u] for (u, _), n in zip(probe['pairs'], probe['negative']))
    report = topology_correction(original, student, [1], probe, .001)
    assert report['gradient_difference_norm'] > 0
    assert report['correction_displacement_norm'] <= report['shared_parameter_norm'] * .001001
    torch.testing.assert_close(student.embedding_dict['user_emb'][1], deleted_row, rtol=0, atol=0)
    for key, value in original_state.items():
        torch.testing.assert_close(original.state_dict()[key], value, rtol=0, atol=0)
    assert all(p.grad is None for p in original.parameters())


def test_same_graph_and_same_probe_has_exact_zero_correction():
    raw, cfg, before, _, original, _ = setup()
    student = compact(original, before, cfg, 'cpu')
    probe = paired_probe(before, [], 991, 1024)
    report = topology_correction(original, student, [], probe, .001)
    assert report['gradient_difference_norm'] == report['correction_displacement_norm'] == 0
    for a, b in zip(original.parameters(), student.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_reset_law_and_only_requested_rows_change():
    raw, cfg, before, after, original, student = setup()
    probe = paired_probe(before, [1], 991, 1024)
    topology_correction(original, student, [1], probe, .001)
    snapshot = {k: v.clone() for k, v in student.state_dict().items()}
    result = reset_isolated_odsr_rows(student, after, [1], 710042)
    bound = math.sqrt(6/(raw.U + cfg['emb_size']))
    assert result['bound'] == bound
    expected = torch.empty(1, 4).uniform_(-bound, bound, generator=torch.Generator().manual_seed(710042))
    torch.testing.assert_close(student.embedding_dict['user_emb'][[1]], expected, rtol=0, atol=0)
    for key, value in student.state_dict().items():
        retained = [0, 2, 3, 4]
        actual = value[retained] if key == 'embedding_dict.user_emb' else value
        old = snapshot[key][retained] if key == 'embedding_dict.user_emb' else snapshot[key]
        torch.testing.assert_close(actual, old, rtol=0, atol=0)


@pytest.mark.parametrize('variant', ['full', 'no_topology', 'uniform_weights', 'no_reset'])
def test_variants_native_retained_training_preserves_isolated_rows_and_resumes(tmp_path, variant):
    raw, cfg, before, after, original, student = setup()
    old_deleted = student.embedding_dict['user_emb'][1].detach().clone()
    weights, metadata = prepare(original, student, before, after, raw, cfg, 42, variant)
    if variant == 'no_reset':
        torch.testing.assert_close(student.embedding_dict['user_emb'][1], old_deleted, rtol=0, atol=0)
    else:
        bound = math.sqrt(6/(raw.U + cfg['emb_size']))
        expected = torch.empty(4).uniform_(-bound, bound, generator=torch.Generator().manual_seed(710042))
        torch.testing.assert_close(student.embedding_dict['user_emb'][1], expected, rtol=0, atol=0)
    if variant == 'uniform_weights':
        assert all(torch.equal(value, torch.ones_like(value)) for value in weights.values())
    isolated = student.embedding_dict['user_emb'][1].detach().clone()
    report = train_retained(student, after, weights, cfg, {'test': variant}, tmp_path, torch.device('cpu'))
    assert report['epoch'] == 2 and report['complete']
    torch.testing.assert_close(student.embedding_dict['user_emb'][1], isolated, rtol=0, atol=0)
    expected = {k: v.clone() for k, v in student.state_dict().items()}
    restored = ODSR(after, cfg, 'cpu')
    resumed = train_retained(restored, after, weights, cfg, {'test': variant}, tmp_path, torch.device('cpu'))
    assert resumed['epoch'] == 2
    for key, value in restored.state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
