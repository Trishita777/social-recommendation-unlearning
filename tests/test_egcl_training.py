"""Check exact dual-optimizer resume and graph-safe user compaction."""
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'src')]
from run_unlearning import graph
from run_egcl_backbone import load_parameters, parameter_state, save_training, resume_training
from social_unlearning.egcl import EGCL


def fixture():
    torch.set_num_threads(2)
    raw = SimpleNamespace(U=4, I=5, train=np.array([[0,0],[1,1],[2,2],[3,3],[1,4]]),
        social=np.array([[0,1],[1,2],[2,3]]), val=np.array([[0,2]]), test=np.array([[1,3]]))
    cfg = dict(embedding_size=4, layer_size=2, contrastive_chunk=2, temperature=.2,
               weight_reg=.1, embedding_reg=.1, intra_reg=.001)
    data = graph(raw, np.arange(4), 'cpu')
    return raw, cfg, data


def opts(model):
    return [torch.optim.Adam(model.weight_parameters(), lr=.01),
            torch.optim.Adam(model.embedding_parameters(), lr=.01)]


def test_two_optimizer_resume_is_exact():
    _, cfg, data = fixture()
    torch.manual_seed(7)
    model = EGCL(data, cfg, 'cpu'); optimizers = opts(model)
    model.alternating_step(*optimizers)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)/'last.pt'
        save_training(path, model, optimizers, cfg, {'source': 'test'}, {'epoch': 1})
        model.alternating_step(*optimizers)
        restored = EGCL(data, cfg, 'cpu'); restored_opts = opts(restored)
        progress = resume_training(path, restored, restored_opts, cfg, {'source': 'test'})
        assert progress['epoch'] == 1
        restored.alternating_step(*restored_opts)
        for name, value in parameter_state(model).items():
            torch.testing.assert_close(parameter_state(restored)[name], value, rtol=0, atol=0)


def test_compaction_preserves_parameters_and_rebuilds_graph():
    raw, cfg, data = fixture()
    model = EGCL(data, cfg, 'cpu')
    keep = np.array([1,2,3]); compact_data = graph(raw, keep, 'cpu')
    compact = EGCL(compact_data, cfg, 'cpu')
    adjacency = compact.recadj.to_dense().clone()
    load_parameters(compact, parameter_state(model), keep)
    torch.testing.assert_close(compact.user_embeddings, model.user_embeddings[keep])
    torch.testing.assert_close(compact.item_embeddings, model.item_embeddings)
    torch.testing.assert_close(compact.recadj.to_dense(), adjacency)
    assert compact.user_num == 3 and compact.recadj.shape == (8,8)
    for loss in compact.objectives():
        assert torch.isfinite(loss)
