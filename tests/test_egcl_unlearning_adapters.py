import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'src')]
from test_edge_deletion import fixture
from run_unlearning import graph
from run_egcl_backbone import load_parameters, parameter_state, train
from run_egcl_unlearning_adapters import train_gnndelete
from social_unlearning.edge_deletion import edge_graph
from social_unlearning.egcl import EGCL
from social_unlearning.egcl_unlearning_adapters import (
    GNNDeleteEGCL, deletion_setup, gnndelete_loss, topology_correction)


SOURCE = ROOT/'external/baseline_sources/GNNDelete'


def models():
    raw, _ = fixture(); cfg = dict(embedding_size=4, layer_size=2, contrastive_chunk=2)
    old_data = graph(raw, np.arange(raw.U), 'cpu'); data = edge_graph(raw, [1], 'cpu')
    original = EGCL(old_data, cfg, 'cpu'); student = EGCL(data, cfg, 'cpu')
    load_parameters(student, parameter_state(original))
    return raw, cfg, data, original, student


def test_native_topology_correction_has_shared_gradient_and_preserves_teacher():
    raw, cfg, data, original, student = models()
    original_state = parameter_state(original)
    report = topology_correction(original, student, [1], .001)
    assert report['blocks']['embedding_attention']['gradient_difference_norm'] > 0
    assert report['blocks']['weight_network']['gradient_difference_norm'] > 0
    assert not torch.count_nonzero(student.user_embeddings[1])
    for name, value in original_state.items():
        torch.testing.assert_close(dict(original.named_parameters())[name], value, rtol=0, atol=0)
    for block in report['blocks'].values():
        assert block['displacement_norm'] <= .001 * block['shared_parameter_norm'] * 1.00001
    # Altered requested scores have no gradients; the correction uses a real shared path.
    users, items = student()
    assert float(users[1].norm()) == 0
    gradients = torch.autograd.grad((users[1] * items[0]).sum(), student.embedding_parameters(), allow_unused=True)
    assert all(g is None or not torch.count_nonzero(g) for g in gradients)


def test_same_graph_correction_is_zero_and_mismatched_parameters_rejected():
    raw, cfg, data, _, student = models()
    original = EGCL(data, cfg, 'cpu'); load_parameters(original, parameter_state(student))
    prior = parameter_state(student)
    report = topology_correction(original, student, [1], .001)
    assert all(block['gradient_difference_norm'] == 0 for block in report['blocks'].values())
    for name, parameter in student.named_parameters():
        expected = prior[name].clone()
        if name == 'user_embeddings':
            expected[1] = 0
        torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match='identical'):
        topology_correction(original, student, [1], .001)


def test_native_candidate_continues_with_full_native_alternating_training(tmp_path):
    raw, cfg, data, original, student = models()
    cfg.update(learning_rate=.001, epochs=2, validation_every=1, min_epochs=2, patience=2)
    student.cfg.update(cfg)
    topology_correction(original, student, [1], .001)
    report = train(student, data, cfg, {'test': 'EGCL-native-candidate'}, tmp_path, torch.device('cpu'))
    assert report['epoch'] == 2 and report['complete']
    assert not torch.count_nonzero(student.user_embeddings[1])
    ck = torch.load(tmp_path/'deployment.pt', weights_only=False)
    restored = EGCL(data, cfg, 'cpu'); load_parameters(restored, ck['model'])
    for x, y in zip(student(), restored()):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    resumed = train(restored, data, cfg, {'test': 'EGCL-native-candidate'}, tmp_path, torch.device('cpu'))
    assert resumed['epoch'] == 2 and resumed['complete']
    for x, y in zip(student(), restored()):
        torch.testing.assert_close(x, y, rtol=0, atol=0)


@pytest.mark.skipif(not SOURCE.exists(), reason='Source DeletionLayer checkout required')
def test_gnndelete_identity_matches_native_and_nodeemb_has_operator_gradient():
    raw, cfg, data, original, student = models()
    masks, forgotten, negative, locality = deletion_setup(raw, [1], student.layers, 7)
    model = GNNDeleteEGCL(student, masks, SOURCE)
    assert len(model.operator_parameters()) == student.layers
    assert model.rec_operators is model.social_operators
    assert all(not p.requires_grad for p in student.parameters())
    with torch.no_grad():
        for p in model.operator_parameters():
            p.copy_(torch.eye(4))
        actual = model(); expected = student()
        for x, y in zip(actual, expected):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
        teacher = tuple(x.detach() for x in model.layer_states(apply=False))
    witness = GNNDeleteEGCL(original, masks, SOURCE, model.rec_operators, model.social_operators)
    loss, report = gnndelete_loss(model, witness, teacher, torch.as_tensor(forgotten),
                                torch.as_tensor(negative), [torch.as_tensor(m) for m in locality])
    loss.backward()
    assert torch.isfinite(loss) and report['randomness'] > 0
    assert any(p.grad is not None and p.grad.norm() > 0 for p in model.operator_parameters())
    assert all(p.grad is None for p in student.parameters())
    assert not torch.count_nonzero(model()[0][1])
    # Official typed negative witnesses never coincide with observed UI/social edges.
    occupied = set(map(tuple, np.concatenate([np.column_stack([raw.train[:, 0], raw.U + raw.train[:, 1]]), raw.social])))
    occupied.update((v, u) for u, v in list(occupied))
    assert not any(tuple(pair) in occupied for pair in negative)


@pytest.mark.skipif(not SOURCE.exists(), reason='Source DeletionLayer checkout required')
def test_gnndelete_two_step_training_serializes_native_parameter_schema(tmp_path):
    raw, cfg, data, original, student = models()
    cfg.update(deletion_steps=2, deletion_validation_every=25, deletion_patience=10,
               request_edge_cap=20, deletion_lr=.001, alpha=.5)
    model, timing = train_gnndelete(original, student, raw, data, cfg, {'test': 'EGCL-adapter'},
                                    tmp_path, SOURCE, torch.device('cpu'), 42)
    assert timing['epoch'] == 2 and (tmp_path/'deployment.pt').exists()
    ck = torch.load(tmp_path/'deployment.pt', weights_only=False)
    masks, _, _, _ = deletion_setup(raw, [1], 2, 42)
    restored = GNNDeleteEGCL(EGCL(data, cfg, 'cpu'), masks, SOURCE)
    load_parameters(restored, ck['model'])
    for x, y in zip(model(), restored()):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    _, _, _, fresh_original, fresh_student = models()
    resumed_model, resumed = train_gnndelete(fresh_original, fresh_student, raw, data, cfg,
                                             {'test': 'EGCL-adapter'}, tmp_path, SOURCE,
                                             torch.device('cpu'), 42)
    assert resumed['epoch'] == 2 and resumed['complete']
    for x, y in zip(model(), resumed_model()):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
