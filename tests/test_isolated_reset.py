import sys
from pathlib import Path
import numpy as np
import pytest
import torch
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'scripts'),str(Path(__file__).resolve().parents[1]/'src')]
from test_edge_deletion import fixture
from run_unlearning import ODSR,graph,loss_terms
from social_unlearning.edge_deletion import edge_graph
from social_unlearning.isolated_reset import reset_isolated_odsr_rows


def test_reset_preserves_retained_predictions_and_initialization_law():
    raw,cfg=fixture();data=edge_graph(raw,[1],'cpu');model=ODSR(data,cfg,'cpu').eval()
    with torch.no_grad():before_u,before_i=model()
    state=torch.get_rng_state().clone()
    meta=reset_isolated_odsr_rows(model,data,[1],991)
    assert torch.equal(torch.get_rng_state(),state)
    with torch.no_grad():after_u,after_i=model()
    active=[0,2,3,4]
    torch.testing.assert_close(after_u[active],before_u[active],rtol=0,atol=0)
    torch.testing.assert_close(after_i,before_i,rtol=0,atol=0)
    expected=torch.empty(1,4).uniform_(-meta['bound'],meta['bound'],generator=torch.Generator().manual_seed(991))
    torch.testing.assert_close(model.embedding_dict['user_emb'][[1]],expected,rtol=0,atol=0)
    u,p=torch.as_tensor(data.train[:4]).T
    loss_terms(model,u,p,torch.tensor([10,11,10,11])).backward()
    assert torch.count_nonzero(model.embedding_dict['user_emb'].grad[1])==0


def test_reset_rejects_nonisolated_graph():
    raw,cfg=fixture();data=graph(raw,np.arange(raw.U),'cpu');model=ODSR(data,cfg,'cpu')
    with pytest.raises(ValueError,match='isolated'):
        reset_isolated_odsr_rows(model,data,[1],991)
