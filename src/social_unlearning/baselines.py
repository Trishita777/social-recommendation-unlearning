"""Explicit ODSR social-user adaptations of selected unlearning baselines.
DA2 uses the checked-out official difficulty estimator and source loss/schedule.
Request-side deleted rows are frozen temporary constants, never deployed.
"""
import importlib.util,math
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.func import functional_call


class RequestGraph:
    def __init__(self,original,student,data,raw,seed):
        self.original=original;self.data=data;self.raw=raw;self.rng=np.random.default_rng(seed)
        self.keep=torch.as_tensor(data.keep,device=student.device);self.frozen_users=original.embedding_dict['user_emb'].detach().clone()
        self.forget_pairs=raw.train[~np.isin(raw.train[:,0],data.keep)]
        self.positive=[set() for _ in range(raw.U)]
        for u,i in raw.train:self.positive[u].add(int(i))
        for p in original.parameters():p.requires_grad_(False)
    def negatives(self,users):
        n=self.rng.integers(self.raw.I,size=len(users))
        for k,u in enumerate(users):
            while int(n[k]) in self.positive[u]:n[k]=self.rng.integers(self.raw.I)
        return n
    def forward(self,student,mode=None):
        self.original.train(student.training);params=dict(student.named_parameters())
        params['embedding_dict.user_emb']=self.frozen_users.index_copy(0,self.keep,student.embedding_dict['user_emb'])
        return functional_call(self.original,params,(mode,))
    def batch(self,student,count=256):
        ids=self.rng.integers(len(self.forget_pairs),size=min(count,len(self.forget_pairs)));pairs=self.forget_pairs[ids]
        u,p=torch.as_tensor(pairs,device=student.device).T;n=torch.as_tensor(self.negatives(pairs[:,0]),device=student.device)
        return ids,u,p,n


def da2_forget_loss(score_good,score_forgotten,difficulty,tau=1.,reduced=True):
    # Source LightGCNFT.loss_fgt + component.util.assess_reweight('coef').
    values=F.softplus(-(score_good-(1+tau*difficulty)*score_forgotten))
    return values.mean() if reduced else values


def da2_eta(grad_norm,embedding_size,eta0=1.,gamma=1.):
    magnitude=grad_norm*math.sqrt(embedding_size)
    eta=eta0*math.exp(-gamma/magnitude) if magnitude>0 else eta0
    return max(0.,min(eta,eta0))


class DA2Social:
    def __init__(self,original,student,data,raw,original_data,source_root,seed,cfg):
        self.request=RequestGraph(original,student,data,raw,seed);self.cfg=cfg;self.eta=cfg.get('eta0',1.);self.last_components={};original.eval();student.eval()
        with torch.no_grad():self.target_u,self.target_i=[x.detach() for x in original()]
        path=Path(source_root)/'assess/base/difficulty.py';spec=importlib.util.spec_from_file_location('official_da2_difficulty',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        u=self.target_u.cpu().numpy();i=self.target_i.cpu().numpy();csr=original_data.ui.tocsr();csc=csr.tocsc();su,si=module.get_xxscore_funcs(u,i)
        values=[module.entanglement(csr,su,si,int(a),int(b),csc) for a,b in self.request.forget_pairs]
        self.difficulty=torch.as_tensor(values,dtype=torch.float32,device=student.device)
        self.stability_u=self.target_u.clone();self.stability_i=self.target_i.clone()
    def __call__(self,student,u,p,n,indices):
        fu,fi=student();av,_=student('adaptive');bv,_=student()
        retain_rec=F.softplus(-(fu[u]*(fi[p]-fi[n])).sum(1)).mean()
        retain_cl=F.cross_entropy(F.normalize(av[u],dim=1)@F.normalize(bv[u],dim=1).T/student.ssl_temp,torch.arange(len(u),device=u.device))
        ids,du,dp,dn=self.request.batch(student);ou,oi=self.request.forward(student)
        sg=(ou[du]*oi[dn]).sum(1);sf=(ou[du]*oi[dp]).sum(1)
        forget=da2_forget_loss(sg,sf,self.difficulty[torch.as_tensor(ids,device=u.device)],self.cfg.get('difficulty_tau',1.))
        # Explicit graph adaptation: retained stability uses rebuilt-graph embeddings;
        # deleted histories use request-local original-graph witnesses only.
        stab=F.mse_loss(torch.cat([fu[u],ou[du]]),self.stability_u[torch.cat([self.request.keep[u],du])])+F.mse_loss(torch.cat([fi[p],fi[n],oi[dp],oi[dn]]),self.stability_i[torch.cat([p,n,dp,dn])])
        repair=retain_rec+student.ssl_reg*retain_cl
        loss=self.eta*forget+(1-self.eta)*repair+self.cfg.get('lambda_stability',1.)*stab
        self.last_components=dict(forget=float(forget.detach()),retain_rec=float(retain_rec.detach()),retain_cl=float(retain_cl.detach()),stability=float(stab.detach()),eta=self.eta,total=float(loss.detach()))
        return loss
    def epoch_end(self,student,epoch):
        # Official source updates after zero-based epoch >0, at delta intervals.
        if epoch<2 or (epoch-1)%self.cfg.get('delta',1)!=0:return
        student.eval();params=list(student.parameters());acc=[torch.zeros_like(p) for p in params];pairs=self.request.forget_pairs;count=len(pairs)
        for start in range(0,count,256):
            batch=pairs[start:start+256];u,p=torch.as_tensor(batch,device=student.device).T;n=torch.as_tensor(self.request.negatives(batch[:,0]),device=student.device)
            ou,oi=self.request.forward(student);sg=(ou[u]*oi[n]).sum(1);sf=(ou[u]*oi[p]).sum(1)
            # Source only_fgt uses unit weights (no difficulty rescaling).
            loss=F.softplus(-(sg-sf)).sum()/count
            grads=torch.autograd.grad(loss,params,allow_unused=True)
            for a,g in zip(acc,grads):
                if g is not None:a.add_(g)
        norm=float(sum(a.square().sum() for a in acc).sqrt());self.eta=da2_eta(norm,student.emb_size,self.cfg.get('eta0',1.),self.cfg.get('gamma',1.))
        with torch.no_grad():
            # Source refreshes stability references at the same schedule boundary.
            ru,ri=student();ou,_=self.request.forward(student);self.stability_u=ou.detach().clone();self.stability_u[self.request.keep]=ru;self.stability_i=ri.detach().clone()
        self.last_components['schedule_gradient_norm']=norm
