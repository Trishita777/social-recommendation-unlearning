import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'src')]
from test_edge_deletion import fixture
from run_controlled_egcl_mio import evaluate_completed, fit_method, load_model, make_graphs
from social_unlearning.edge_deletion import filtered_raw
from social_unlearning.egcl import EGCL
from social_unlearning.benchmark_mio import BenchmarkMIO


def setup():
    raw, _ = fixture(); raw.meta = dict(seed=42, dataset='toy')
    deleted = np.array([1]); absent = np.array([3])
    before, after = make_graphs(raw, deleted, absent, 'cpu')
    cfg = dict(seed=42, embedding_size=4, layer_size=2, contrastive_chunk=2,
               learning_rate=.001, epochs=2, min_epochs=2, validation_every=1, patience=2,
               correction_relative_step=.001, deletion_steps=2, deletion_validation_every=25,
               deletion_patience=10, deletion_lr=.001, request_edge_cap=20, alpha=.5)
    return raw, deleted, absent, before, after, cfg


def test_native_never_member_graph_rows_and_full_candidate_objective():
    raw, deleted, absent, before, after, cfg = setup()
    assert 1 in before.train[:, 0] and 3 not in before.train[:, 0]
    assert not np.isin(after.train[:, 0], [1, 3]).any()
    old = EGCL(before, cfg, 'cpu'); new = EGCL(after, cfg, 'cpu')
    assert old.user_embeddings.shape == new.user_embeddings.shape == (raw.U, 4)
    for model, excluded in [(old, [3]), (new, [1, 3])]:
        assert not torch.count_nonzero(model()[0][excluded])
        for adjacency in [model.recadj, model.socadj]:
            assert not torch.isin(adjacency.coalesce().indices(), torch.as_tensor(excluded)).any()
        _, loss = model.objectives()
        gradient = torch.autograd.grad(loss, model.user_embeddings)[0]
        # Empty rows receive native global L2 gradients, despite no history edges.
        torch.testing.assert_close(gradient[excluded], 2 * model.user_embeddings[excluded], rtol=1e-5, atol=1e-7)


def test_controlled_native_methods_resume_and_report_two_classes(tmp_path):
    raw, deleted, absent, before, after, cfg = setup()
    raw_members = filtered_raw(raw, absent); provenance = {'test': 'controlled-EGCL'}
    base = tmp_path/'targets'; base.mkdir()
    bank = tmp_path/'bank'; bank.mkdir()
    torch.save(dict(model=BenchmarkMIO(8).state_dict()), bank/'attacker.pt')
    for method in ['original', 'retrain', 'graph_only', 'retain_finetune', 'native_topology']:
        fit_method(method, base, raw, before, after, raw_members, deleted, cfg, provenance, torch.device('cpu'))
        # A completed rerun must reuse without training or altering its checkpoint.
        old = (base/method/'deployment.pt').read_bytes()
        fit_method(method, base, raw, before, after, raw_members, deleted, cfg, provenance, torch.device('cpu'))
        assert (base/method/'deployment.pt').read_bytes() == old
    evaluate_completed(base, tmp_path/'results', raw, raw_members, before, after, deleted,
                       absent, bank, provenance, torch.device('cpu'), 5)
    report = json.loads((tmp_path/'results/results.json').read_text())
    assert report['original_membership_labels'] == [1, 0]
    assert len(report['rows']) == 5
    assert set(report['unavailable_methods']) == {'scif', 'da2'}
    for row in report['rows']:
        assert row['metrics_at_0_5']['members'] == row['metrics_at_0_5']['nonmembers'] == 1
        assert row['mean_native_user_norm']['never_members'] == 0
        if row['method'] != 'original':
            assert row['mean_native_user_norm']['deleted_members'] == 0


@pytest.mark.skipif(not (ROOT/'external/baseline_sources/GNNDelete').exists(), reason='Official source required')
def test_controlled_gnndelete_uses_never_member_excluded_original(tmp_path):
    raw, deleted, absent, before, after, cfg = setup()
    raw_members = filtered_raw(raw, absent); prov = {'test': 'controlled-EGCL-GNNDelete'}
    base = tmp_path/'targets'; base.mkdir()
    for method in ['original', 'gnndelete_native']:
        fit_method(method, base, raw, before, after, raw_members, deleted, cfg, prov, torch.device('cpu'))
    model, ck = load_model(base/'gnndelete_native', after, raw_members, deleted, torch.device('cpu'))
    assert not torch.count_nonzero(model()[0][[1, 3]])
    assert ck['training']['epoch'] == 2
    assert ck['provenance']['method'] == 'gnndelete_native'
