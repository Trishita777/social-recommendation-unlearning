"""Declared DiffNet++ adaptations of source-audited unlearning baselines.

SCIF/DA2/RRL preserve original propagation and persistent IDs, as in the active
ODSR comparison. GNNDelete uses original witnesses and compact retained
reference/inference. These are social-backbone adaptations, not exact published
benchmark reproductions. Native DiffNet++ repair uses its released MSE loss.
"""
import importlib.util
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F
from .baselines import da2_forget_loss,da2_eta
from .gnndelete_edges import deletion_class,neighborhoods


class GNNDeleteDiffNet(nn.Module):
    def __init__(self,body,masks,source,operators=None):
        super().__init__();self.body=body
        for p in body.parameters():p.requires_grad_(False)
        layer=deletion_class(source);device=body.user_embeddings.device
        self.operators=operators if operators is not None else nn.ModuleList([layer(body.dimension,None).to(device) for _ in range(2)])
        for k,mask in enumerate(masks):self.register_buffer(f'mask_{k}',torch.as_tensor(mask,dtype=torch.bool,device=device))

    def forward(self,return_layers=False,apply=True):
        b=self.body;u,i=b.user_embeddings,b.item_embeddings;us=[];items=[];states=[]
        for k in range(2):
            ui=b.aggregate(k,'ui',i,len(u));uu=b.aggregate(k,'uu',u,len(u));iu=b.aggregate(k,'iu',u,len(i))
            a=b.high[f'l{k}_ui'](torch.cat([u,ui],1))+.7;c=b.high[f'l{k}_uu'](torch.cat([u,uu],1))+.3
            un=.5*u+.5*(a*ui+c*uu)/(a+c)
            a=b.high[f'l{k}_self'](i)+1.;c=b.high[f'l{k}_iu'](iu)+1.
            it=(a*i+c*iu)/(a+c)
            joined=torch.cat([un,it])
            if apply:joined=self.operators[k](joined,getattr(self,f'mask_{k}'))
            u,i=joined[:len(u)],joined[len(u):];us.append(u);items.append(i);states.append(joined)
        if return_layers:return states
        return torch.cat(us+[b.user_embeddings],1),torch.cat(items+[b.item_embeddings],1)


def deletion_targets(raw,deleted,data,seed):
    masks,forgotten,edges=neighborhoods(raw,deleted,2)
    rng=np.random.default_rng(seed);forget=forgotten[rng.choice(len(forgotten),min(4096,len(forgotten)),replace=False)]
    occupied=set(map(tuple,edges));occupied.update((int(b),int(a)) for a,b in edges)
    negative=[]
    for _,v in forget:
        while True:
            u=int(rng.choice(data.keep));w=int(rng.integers(raw.U,raw.U+raw.I)) if v>=raw.U else int(rng.choice(data.keep))
            if u!=w and (u,w) not in occupied:negative.append((u,w));break
    selection=np.concatenate([data.keep,np.arange(raw.U,raw.U+raw.I)])
    mapping=np.full(raw.U+raw.I,-1,dtype=np.int64);mapping[selection]=np.arange(len(selection))
    negative=mapping[np.asarray(negative)]
    assert (negative>=0).all()
    outside=np.ones(raw.U+raw.I,dtype=bool);outside[np.unique(forgotten)]=False
    return masks,[m[selection] for m in masks],forget,negative,[(m&outside)[selection] for m in masks]


class DA2DiffNet:
    def __init__(self,original,student,raw,deleted,source,seed):
        self.student=student;self.rng=np.random.default_rng(seed);self.raw=raw;self.eta=1.
        self.pairs=raw.train[np.isin(raw.train[:,0],deleted)]
        self.positive=[set() for _ in range(raw.U)]
        for u,i in raw.train:self.positive[u].add(int(i))
        original.eval()
        with torch.no_grad():self.ref_u,self.ref_i=[v.detach().clone() for v in original()]
        path=Path(source)/'assess/base/difficulty.py';spec=importlib.util.spec_from_file_location('diffnet_da2_difficulty',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        csr=sp.csr_matrix((np.ones(len(raw.train)),(raw.train[:,0],raw.train[:,1])),shape=(raw.U,raw.I));csc=csr.tocsc()
        su,si=module.get_xxscore_funcs(self.ref_u.cpu().numpy(),self.ref_i.cpu().numpy())
        self.difficulty=torch.tensor([module.entanglement(csr,su,si,int(u),int(i),csc) for u,i in self.pairs],device=student.user_embeddings.device,dtype=torch.float32)
        self.last_components={}

    def negatives(self,users):
        result=self.rng.integers(self.raw.I,size=len(users))
        for k,u in enumerate(users):
            while int(result[k]) in self.positive[u]:result[k]=self.rng.integers(self.raw.I)
        return result

    def loss(self,batch):
        u,i=self.student();users,items,labels=batch
        repair=.5*(labels-(u[users]*i[items]).sum(1).sigmoid()).square().mean()
        ids=self.rng.integers(len(self.pairs),size=min(256,len(self.pairs)));pairs=self.pairs[ids]
        du,dp=torch.as_tensor(pairs,device=u.device).T;dn=torch.as_tensor(self.negatives(pairs[:,0]),device=u.device)
        forget=da2_forget_loss((u[du]*i[dn]).sum(1),(u[du]*i[dp]).sum(1),self.difficulty[torch.as_tensor(ids,device=u.device)])
        # Match the source stability term on retained/forgotten sampled endpoints.
        stability=F.mse_loss(torch.cat([u[users],u[du]]),self.ref_u[torch.cat([users,du])])
        stability+=F.mse_loss(torch.cat([i[items],i[dp],i[dn]]),self.ref_i[torch.cat([items,dp,dn])])
        result=self.eta*forget+(1-self.eta)*repair+stability
        self.last_components=dict(forget=float(forget.detach()),repair=float(repair.detach()),stability=float(stability.detach()),eta=self.eta)
        return result

    def epoch_end(self,epoch):
        if epoch<2:return
        u,i=self.student();du,dp=torch.as_tensor(self.pairs,device=u.device).T
        dn=torch.as_tensor(self.negatives(self.pairs[:,0]),device=u.device)
        loss=F.softplus(-((u[du]*i[dn]).sum(1)-(u[du]*i[dp]).sum(1))).mean()
        grads=torch.autograd.grad(loss,tuple(self.student.parameters()),allow_unused=True)
        magnitude=float(sum(g.square().sum() for g in grads if g is not None).sqrt())
        self.eta=da2_eta(magnitude,self.student.dimension)
        with torch.no_grad():self.ref_u,self.ref_i=[v.detach().clone() for v in self.student()]
        self.last_components['schedule_gradient_norm']=magnitude
