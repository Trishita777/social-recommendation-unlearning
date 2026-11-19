"""SCIF/DA2 social adaptations for native EGCL, fixed-ID incident-edge deletion.
Not unchanged published reproductions. Native EGCL repair is never replaced by BPR.
"""
import importlib.util
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.func import functional_call
from social_unlearning.egcl import EGCL
from social_unlearning.baselines import da2_forget_loss, da2_eta


class ProbeEGCL(EGCL):
    """Fixed anchor subsample, full candidate denominator, unbiased sum scaling."""
    def set_probe(self, seed, size=512):
        rng = np.random.default_rng(seed)
        self.probes = {n: torch.as_tensor(np.sort(rng.choice(n, min(n,size), replace=False)),
                         device=self.device) for n in {self.user_num,self.item_num}}
    def info_nce(self, a, b, anchor_weights=None):
        ids=self.probes[len(a)]
        # Clamp squared norm before sqrt: equivalent values/first derivatives,
        # finite double backward for structurally zero isolated-node vectors.
        normalize=lambda x: x/x.square().sum(1,keepdim=True).clamp_min(1e-12).sqrt()
        a=normalize(a); b=normalize(b)
        values=F.cross_entropy(a[ids]@b.T/self.cfg.get('temperature',.05),ids,reduction='none')
        if anchor_weights is not None:values=values*anchor_weights[ids].detach()
        return values.sum()*(len(a)/len(ids))


def scif_rhs(original, student, normalizer):
    for model in (original,student):
        model.eval()
        for p in model.parameters():p.requires_grad_(False)
        model.user_embeddings.requires_grad_(True)
    grads=[]
    for model in (original,student):
        loss=model.objectives()[1]/normalizer
        grads.append(torch.autograd.grad(loss,model.user_embeddings)[0].detach())
    original.user_embeddings.requires_grad_(False)
    return grads[1]-grads[0]


class DA2EGCL:
    def __init__(self, original, student, raw, deleted, original_data, source, seed, cfg):
        self.original=original; self.student=student; self.cfg=cfg
        self.rng=np.random.default_rng(seed); self.deleted=np.asarray(deleted)
        self.keep=np.setdiff1d(np.arange(raw.U),deleted)
        self.pairs=raw.train[np.isin(raw.train[:,0],deleted)]
        if not len(self.pairs):raise ValueError('No forgotten interactions')
        self.positive=[set() for _ in range(raw.U)]
        for u,i in raw.train:self.positive[u].add(int(i))
        self.item_num=raw.I; self.eta=1.; self.last_components={}
        original.eval()
        for p in original.parameters():p.requires_grad_(False)
        with torch.no_grad():self.ref_u,self.ref_i=[x.detach().clone() for x in original()]
        path=Path(source)/'assess/base/difficulty.py'
        spec=importlib.util.spec_from_file_location('official_da2_difficulty',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        su,si=module.get_xxscore_funcs(self.ref_u.cpu().numpy(),self.ref_i.cpu().numpy())
        csr=original_data.ui.tocsr();csc=csr.tocsc()
        self.difficulty=torch.tensor([module.entanglement(csr,su,si,int(u),int(i),csc)
                                     for u,i in self.pairs],device=student.device,dtype=torch.float32)
    def negatives(self, users):
        values=self.rng.integers(self.item_num,size=len(users))
        for j,u in enumerate(users):
            if len(self.positive[u])>=self.item_num:raise ValueError('No negative items')
            while int(values[j]) in self.positive[u]:values[j]=self.rng.integers(self.item_num)
        return values
    def witness(self):
        # Keep persistent IDs, as in existing ODSR edge-user DA2 adaptation.
        # Shared current parameters run through original graph buffers only here.
        return functional_call(self.original,dict(self.student.named_parameters()),())
    def forgetting(self, difficulty=True):
        u,p=torch.as_tensor(self.pairs,device=self.student.device).T
        n=torch.as_tensor(self.negatives(self.pairs[:,0]),device=self.student.device)
        ou,oi=self.witness();good=(ou[u]*oi[n]).sum(1);bad=(ou[u]*oi[p]).sum(1)
        loss=da2_forget_loss(good,bad,self.difficulty if difficulty else torch.zeros_like(bad))
        return loss,(u,p,n,ou,oi)
    def loss(self, data):
        # Mean-normalize native full-graph embedding objective to match mean forget loss.
        repair=self.student.objectives()[1]/max(1,len(data.train))
        forget,(du,dp,dn,ou,oi)=self.forgetting()
        ru,ri=self.student()
        ids=self.rng.choice(len(data.train),min(2048,len(data.train)),replace=False)
        u,p=torch.as_tensor(data.train[ids],device=self.student.device).T
        n=torch.as_tensor(self.negatives(data.train[ids,0]),device=self.student.device)
        stability=F.mse_loss(torch.cat([ru[u],ou[du]]),self.ref_u[torch.cat([u,du])])
        stability+=F.mse_loss(torch.cat([ri[p],ri[n],oi[dp],oi[dn]]),self.ref_i[torch.cat([p,n,dp,dn])])
        loss=self.eta*forget+(1-self.eta)*repair+stability
        self.last_components=dict(forget=float(forget.detach()),repair=float(repair.detach()),
                                  stability=float(stability.detach()),eta=self.eta)
        return loss
    def epoch_end(self, epoch):
        if epoch<2:return
        self.student.eval();loss,_=self.forgetting(difficulty=False)
        params=self.student.embedding_parameters()
        grads=torch.autograd.grad(loss,params,allow_unused=True)
        norm=float(sum(g.square().sum() for g in grads if g is not None).sqrt())
        self.eta=da2_eta(norm,self.student.user_embeddings.shape[1])
        with torch.no_grad():
            ru,ri=self.student();ou,_=self.witness()
            self.ref_u=ou.detach().clone();self.ref_u[self.keep]=ru[self.keep]
            self.ref_i=ri.detach().clone()
        self.last_components['schedule_gradient_norm']=norm
