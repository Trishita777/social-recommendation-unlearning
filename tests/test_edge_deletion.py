import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from run_unlearning import graph,ODSR,compact
from social_unlearning.edge_deletion import edge_graph,edge_sensitivity,EdgeGuardedForgetRetain
from social_unlearning.egcl import EGCL


def fixture():
    torch.set_num_threads(2);torch.manual_seed(4)
    raw=SimpleNamespace(U=5,I=12,train=np.array([[u,2*u+i] for u in range(5) for i in range(2)]),
        social=np.array([[0,1],[1,0],[1,2],[3,1],[2,3],[3,4]]),
        val=np.array([[0,11],[1,10],[2,10]]),test=np.array([[0,10],[1,11],[2,11]]))
    cfg=dict(emb_size=4,n_layer=2,reg=0.,eps=.46,ssl_reg=.18,tau=.2,dropout=0.)
    return raw,cfg


def test_incident_edges_both_directions_removed_and_ids_preserved():
    raw,cfg=fixture();data=edge_graph(raw,[1],torch.device('cpu'))
    assert data.user_num==raw.U and data.item_num==raw.I
    np.testing.assert_array_equal(data.keep,np.arange(raw.U))
    np.testing.assert_array_equal(data.mapping,np.arange(raw.U))
    assert not (data.train[:,0]==1).any() and not (data.social==1).any()
    assert data.ui.shape==(5,12) and data.ui.getrow(1).nnz==0
    a=data.adj_uu.to_dense();b=data.norm_adj.to_dense()
    assert not a[1].any() and not a[:,1].any() and not b[1].any() and not b[:,1].any()
    # Recompute normalized bipartite adjacency independently: zeroing an old
    # normalized matrix would give the wrong degrees for remaining neighbors.
    ui=data.ui.toarray();full=np.block([[np.zeros((5,5)),ui],[ui.T,np.zeros((12,12))]])
    degree=full.sum(1);inv=np.zeros_like(degree);np.power(degree,-.5,out=inv,where=degree>0)
    np.testing.assert_allclose(b.numpy(),inv[:,None]*full*inv[None,:],rtol=1e-6)
    assert data.counts['removed_users']==0 and data.counts['requested_users']==1
    assert data.counts['removed_interactions']==2 and data.counts['removed_directed_social_edges']==4
    original=ODSR(graph(raw,np.arange(5),'cpu'),cfg,'cpu')
    student=compact(original,data,cfg,'cpu')
    torch.testing.assert_close(student.embedding_dict['user_emb'],original.embedding_dict['user_emb'])
    assert student()[0].shape==(5,4)


def test_forgetting_uses_altered_graph_and_updates_requested_row():
    raw,cfg=fixture();device=torch.device('cpu');data=edge_graph(raw,[1],device)
    original=ODSR(graph(raw,np.arange(5),device),cfg,device)
    student=compact(original,data,cfg,device)
    weights=edge_sensitivity(original,student,data,4)
    objective=EdgeGuardedForgetRetain(original,student,data,raw,weights,4)
    student.eval();a,b=objective.request_forward(student);c,d=student()
    torch.testing.assert_close(a,c);torch.testing.assert_close(b,d)
    u,p=torch.as_tensor(data.train[:4]).T;n=torch.tensor([10,11,10,11])
    initial=student.embedding_dict['user_emb'][1].detach().clone()
    objective.backward_step(student,u,p,n,torch.arange(4),list(student.parameters()),True,2)
    gradient=student.embedding_dict['user_emb'].grad
    assert torch.isfinite(gradient).all() and gradient[1].norm()>0
    torch.optim.Adam(student.parameters(),lr=.0001).step()
    assert not torch.equal(initial,student.embedding_dict['user_emb'][1])
    assert all(p.grad is None for p in original.parameters())


def test_egcl_rebuilds_native_full_size_graphs():
    raw,_=fixture();data=edge_graph(raw,[1],'cpu')
    model=EGCL(data,dict(embedding_size=4,layer_size=2,contrastive_chunk=2),'cpu')
    assert model.user_embeddings.shape==(5,4)
    for a in [model.recadj.to_dense(),model.socadj.to_dense()]:
        assert not a[1].any() and not a[:,1].any()
    assert torch.isfinite(model()[0]).all()


def test_baseline_adapters_keep_requested_rows_and_use_correct_graphs(tmp_path):
    import pytest
    from social_unlearning.edge_deletion import edge_da2
    from social_unlearning.scif_edges import gradient_change
    from social_unlearning.gnndelete_edges import train_deletion
    if not (ROOT/'external/baseline_sources/DA2Unlearn').exists():
        pytest.skip('Official baseline source checkouts are on DGL')
    raw,cfg=fixture();device=torch.device('cpu');data=edge_graph(raw,[1],device)
    orig=graph(raw,np.arange(raw.U),device);original=ODSR(orig,cfg,device)
    student=compact(original,data,cfg,device)
    objective=edge_da2(original,student,data,raw,orig,ROOT/'external/baseline_sources/DA2Unlearn',4,cfg)
    student.eval();a,b=objective.request.forward(student);c,d=student()
    torch.testing.assert_close(a,c);torch.testing.assert_close(b,d)
    assert len(objective.request.forget_pairs)==2
    u,p=torch.as_tensor(data.train[:4]).T;n=torch.tensor([10,11,10,11])
    loss=objective(student,u,p,n,np.arange(4));loss.backward()
    assert torch.isfinite(student.embedding_dict['user_emb'].grad).all()
    assert student.embedding_dict['user_emb'].grad[1].norm()>0
    student=compact(original,data,cfg,device)
    rhs,probe,metadata=gradient_change(original,student,data,raw,4,4)
    assert rhs.shape==(raw.U,4) and torch.isfinite(rhs).all()
    student,info=train_deletion(original,raw,data,cfg,tmp_path,device,4,{'test':'edge'},max_steps=1)
    assert student.user_num==raw.U and student()[0].shape==(raw.U,4)
    assert not student.body.adj_uu.to_dense()[1].any()
