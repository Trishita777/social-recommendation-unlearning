import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'src'),str(Path(__file__).resolve().parents[1]/'scripts')]
from social_unlearning.influence_spillover import cg, SurvivingObjective, solve_influence


def test_cg_dense_reference_and_negative_curvature():
    h=torch.tensor([[4.,1.],[1.,3.]],dtype=torch.double);b=torch.tensor([1.,2.],dtype=torch.double)
    x,r=cg(lambda v:(h@v[0],),(b,),tolerance=1e-10,max_steps=5)
    assert r['converged']
    torch.testing.assert_close(x[0],torch.linalg.solve(h,b))
    _,r=cg(lambda v:(-v[0],),(b,))
    assert not r['converged'] and r['reason']=='nonpositive_curvature'


class Quadratic(torch.nn.Module):
    def __init__(self,users,center,scale):
        super().__init__();self.embedding_dict=torch.nn.ParameterDict({'user_emb':torch.nn.Parameter(torch.arange(users,dtype=torch.double)[:,None]+1.)})
        self.shared=torch.nn.Parameter(torch.tensor([.4],dtype=torch.double));self.center=center;self.scale=scale
    def compute_loss(self,*batch):
        u=self.embedding_dict['user_emb'];return self.scale*(u.square().sum()+self.shared.square().sum())/2-self.center*self.shared.sum()+self.shared*u.sum()/10


def test_surviving_coordinates_native_discrepancy_and_dense_hessian():
    old=Quadratic(4,.3,2.);new=Quadratic(3,.8,1.)
    keep=np.array([0,2,3])
    with torch.no_grad():new.embedding_dict['user_emb'].copy_(old.embedding_dict['user_emb'][keep])
    original={k:v.detach().clone() for k,v in old.named_parameters()}
    objective=SurvivingObjective(old,new,keep,'ODSR',(),(),seed=42)
    b,stats=objective.discrepancy();oldparams=objective.parameters_for(True)
    assert torch.equal(oldparams['model.embedding_dict.user_emb'][1],original['embedding_dict.user_emb'][1])
    assert sum(v.numel() for v in objective.params)==4
    vector=tuple(torch.randn_like(p) for p in objective.params)
    hv=objective.hvp(vector)
    # Dense Hessian in explicitly surviving coordinates includes shared cross derivatives.
    h=torch.eye(4,dtype=torch.double)*2;h[0,1:]=.1;h[1:,0]=.1
    flat=lambda v:torch.cat([x.reshape(-1) for x in v])
    torch.testing.assert_close(flat(hv),h@flat(vector))
    delta,report=solve_influence(objective,damping=.2,tolerance=1e-10,max_steps=10)
    torch.testing.assert_close(flat(delta),torch.linalg.solve(h+.2*torch.eye(4,dtype=torch.double),flat(b)))
    assert report['solver']['converged']
    for n,p in old.named_parameters():torch.testing.assert_close(p,original[n])


def test_native_odsr_hvp_replay_and_frozen_deleted_rows():
    from run_unlearning import graph,ODSR
    raw=SimpleNamespace(U=4,I=5,train=np.array([[0,0],[0,1],[1,1],[1,2],[2,2],[2,3],[3,3],[3,4]]),social=np.array([[0,1],[1,2],[2,3]]),val=np.array([[0,2],[2,4]]),test=np.array([[0,3],[2,0]]))
    device=torch.device('cpu');olddata=graph(raw,np.arange(4),device);keep=np.array([0,2,3]);newdata=graph(raw,keep,device)
    cfg=dict(emb_size=3,n_layer=1,reg=.001,eps=.1,ssl_reg=.05,dropout=.1)
    old=ODSR(olddata,cfg,device).double();new=ODSR(newdata,cfg,device).double()
    for model in [old,new]:
        for name in ['adj_ui','adj_uu','sparse_norm_adj']:setattr(model,name,getattr(model,name).double())
    state=old.state_dict();state['embedding_dict.user_emb']=state['embedding_dict.user_emb'][keep];new.load_state_dict(state)
    batch=tuple(torch.tensor(x) for x in [[0,1,2],[0,1,2],[4,4,0]])
    rb=tuple(torch.tensor(x) for x in [[0,1,2],[0,2,3],[4,0,1]])
    obj=SurvivingObjective(old,new,keep,'ODSR',batch,rb,seed=9)
    assert float(obj.loss(True))==float(obj.loss(True))
    v=tuple(torch.randn_like(p) for p in obj.params)
    a=obj.hvp(v);b=obj.hvp(v)
    for x,y in zip(a,b):
        assert torch.isfinite(x).all();torch.testing.assert_close(x,y,rtol=0,atol=0)


def test_egcl_blocks_preserve_native_losses_and_hvp():
    from run_unlearning import graph
    from social_unlearning.egcl_curvature import StreamedCurvatureEGCL as EGCL
    from run_egcl_backbone import load_parameters,parameter_state
    raw=SimpleNamespace(U=4,I=6,train=np.array([[0,0],[0,1],[1,1],[1,2],[2,2],[2,3],[3,3],[3,4]]),social=np.array([[0,1],[1,2],[2,3]]),val=np.array([[0,2],[2,4]]),test=np.array([[0,3],[2,0]]))
    device=torch.device('cpu');keep=np.array([0,2,3]);olddata=graph(raw,np.arange(4),device);newdata=graph(raw,keep,device)
    cfg=dict(embedding_size=3,layer_size=1,contrastive_chunk=2,temperature=.2)
    old=EGCL(olddata,cfg,device);new=EGCL(newdata,cfg,device)
    state=parameter_state(old);state['user_embeddings']=state['user_embeddings'][keep];load_parameters(new,state)
    for block,index in [('embedding',1),('weight',0)]:
        obj=SurvivingObjective(old,new,keep,'EGCL',seed=7,block=block)
        torch.testing.assert_close(obj.loss(False),new.objectives()[index])
        grad=obj.gradient(False)[1]
        direct=torch.autograd.grad(new.objectives()[index],obj.params)
        for a,b in zip(grad,direct):torch.testing.assert_close(a,b)
        v=tuple(torch.randn_like(p) for p in obj.params)
        hv=obj.hvp(v)
        assert all(torch.isfinite(x).all() for x in hv)


def test_egcl_zero_view_normalization_preserves_native_value_and_gradient():
    from social_unlearning.egcl_curvature import finite_normalize
    from torch.nn import functional as F
    x=torch.tensor([[0.,0.,0.],[.2,.4,.1],[1e-9,-1e-9,0.]],dtype=torch.double,requires_grad=True)
    native=F.normalize(x,dim=1,eps=1e-6);safe=finite_normalize(x)
    torch.testing.assert_close(native,safe,rtol=1e-12,atol=1e-12)
    v=torch.randn_like(x)
    gn=torch.autograd.grad((native*v).sum(),x)[0]
    gs=torch.autograd.grad((safe*v).sum(),x,create_graph=True)[0]
    torch.testing.assert_close(gn,gs,rtol=1e-12,atol=1e-12)
    h=torch.autograd.grad(gs.square().sum(),x)[0]
    assert torch.isfinite(h).all()


def test_streamed_native_contrast_value_gradient_and_hessian():
    from social_unlearning.egcl_curvature import streamed_info_nce
    from torch.nn import functional as F
    torch.manual_seed(22)
    a=torch.randn(5,3,dtype=torch.double,requires_grad=True);b=torch.randn(5,3,dtype=torch.double,requires_grad=True)
    weights=torch.rand(5,dtype=torch.double);temp=.7
    native=(F.cross_entropy(a@b.T/temp,torch.arange(5),reduction='none')*weights).sum()
    streamed=streamed_info_nce(a,b,temp,2,weights)
    torch.testing.assert_close(streamed,native,rtol=1e-12,atol=1e-12)
    ng=torch.autograd.grad(native,(a,b),create_graph=True);sg=torch.autograd.grad(streamed,(a,b),create_graph=True)
    for x,y in zip(ng,sg):torch.testing.assert_close(x,y,rtol=1e-11,atol=1e-11)
    direction=(torch.randn_like(a),torch.randn_like(b))
    nh=torch.autograd.grad(sum((x*d).sum() for x,d in zip(ng,direction)),(a,b))
    sh=torch.autograd.grad(sum((x*d).sum() for x,d in zip(sg,direction)),(a,b))
    for x,y in zip(nh,sh):torch.testing.assert_close(x,y,rtol=1e-10,atol=1e-10)
    assert torch.autograd.gradcheck(lambda x,y:streamed_info_nce(x,y,temp,2),(a,b))
    assert torch.autograd.gradgradcheck(lambda x,y:streamed_info_nce(x,y,temp,2),(a,b))


def test_shifted_krylov_reuses_hvps_and_matches_dense_indefinite_system():
    from social_unlearning.influence_spillover import solve_shifted_krylov
    class Objective:
        params=[torch.ones(3,dtype=torch.double)]
        h=torch.tensor([[-2.,.2,0.],[.2,3.,.5],[0.,.5,4.]],dtype=torch.double)
        rhs=torch.tensor([1.,2.,-1.],dtype=torch.double)
        calls=0
        def discrepancy(self):return (self.rhs,),{}
        def hvp(self,v):self.calls+=1;return (self.h@v[0],)
    o=Objective();delta,report=solve_shifted_krylov(o,damping=.01,tolerance=1e-10,max_steps=6,max_restarts=5)
    mu=report['solver']['damping'];assert mu==10.
    torch.testing.assert_close(delta[0],torch.linalg.solve(o.h+mu*torch.eye(3,dtype=torch.double),o.rhs),rtol=1e-10,atol=1e-10)
    assert o.calls<=4 and report['solver']['relative_residual']<1e-10
