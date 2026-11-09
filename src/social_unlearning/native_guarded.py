"""ODSR-native weighted forgetting/retaining objective.
Deleted users occur only in ephemeral request-side original-graph forwards.
The active student, retained losses, candidate negatives, and final artifact
contain retained user rows only. No old-graph teacher is deployed.
"""
import numpy as np
import torch
from torch.nn import functional as F
from torch.func import functional_call


def normalized_weights(x):
    x=x.detach(); w=1+(x/x.mean().clamp_min(1e-8)).clamp(0,4); return w/w.mean()


def weighted_infonce(anchor,positive,candidates,temperature,weights=None):
    """Same log-softmax InfoNCE form, with an explicit positive and retained negatives."""
    a=F.normalize(anchor,dim=1); p=F.normalize(positive,dim=1); c=F.normalize(candidates,dim=1)
    positive_logits=(a*p).sum(1)/temperature
    logits=torch.cat([positive_logits[:,None],a@c.T/temperature],dim=1)
    values=torch.logsumexp(logits,dim=1)-positive_logits
    return values.mean() if weights is None else (values*weights.detach()).mean()


class GuardedNativeForgetRetain:
    def __init__(self,original,student,data,raw,weights,seed,*,forget_strength=.1,forget_cl_strength=.1,uniform=False,contrastive_weighting=True,forget_batch=256):
        self.original=original; self.data=data; self.raw=raw; self.weights=weights; self.uniform=uniform; self.contrastive_weighting=contrastive_weighting; self.rng=np.random.default_rng(seed)
        self.forget_strength=forget_strength; self.forget_cl_strength=forget_cl_strength; self.forget_batch=forget_batch
        device=student.device; self.keep=torch.as_tensor(data.keep,device=device); deleted=np.setdiff1d(np.arange(raw.U),data.keep)
        self.deleted=torch.as_tensor(deleted,device=device); self.forget_pairs=raw.train[np.isin(raw.train[:,0],deleted)]
        self.full_positive=[set() for _ in range(raw.U)]
        for u,i in raw.train:self.full_positive[u].add(int(i))
        for p in original.parameters():p.requires_grad_(False)
        original.eval()
        with torch.no_grad():
            ou,oi=original(); ru,ri=student(); self.initial_deleted=original.embedding_dict['user_emb'].detach().clone()
            # Counterfactual isolated deleted user: only layer-zero pooling remains.
            isolated=original.alphas.softmax(0)[0]*original.embedding_dict['user_emb']
            neg=self.sample_negatives(self.forget_pairs[:,0]); all_weights=[]
            for start in range(0,len(neg),4096):
                u,p=torch.as_tensor(self.forget_pairs[start:start+4096],device=device).T; n=torch.as_tensor(neg[start:start+4096],device=device)
                old=(ou[u]*(oi[p]-oi[n])).sum(1); counter=(isolated[u]*(ri[p]-ri[n])).sum(1)
                all_weights.append((old-counter).abs())
            self.forget_rec=normalized_weights(torch.cat(all_weights))
            # Per-anchor distribution sensitivity on common retained candidate IDs.
            candidates=self.keep[torch.as_tensor(self.rng.choice(len(self.keep),min(512,len(self.keep)),replace=False),device=device)]
            old=F.normalize(ou[self.deleted],dim=1)@F.normalize(ou[candidates],dim=1).T/original.ssl_temp
            candidate_local=torch.as_tensor(data.mapping[candidates.cpu().numpy()],device=device)
            new=F.normalize(isolated[self.deleted],dim=1)@F.normalize(ru[candidate_local],dim=1).T/original.ssl_temp
            po=old.softmax(1); pn=new.softmax(1); mean=(po+pn)/2
            js=.5*((po*(po.clamp_min(1e-12).log()-mean.clamp_min(1e-12).log())).sum(1)+(pn*(pn.clamp_min(1e-12).log()-mean.clamp_min(1e-12).log())).sum(1))
            aw=torch.ones(raw.U,device=device); aw[self.deleted]=normalized_weights(js); self.forget_anchor=aw
        self.last_components={}
        self.components={}
    def sample_negatives(self,users):
        n=self.rng.integers(self.raw.I,size=len(users))
        for k,u in enumerate(users):
            while int(n[k]) in self.full_positive[u]:n[k]=self.rng.integers(self.raw.I)
        return n
    def request_forward(self,student,mode=None):
        # Differentiable lift of compact rows; deleted rows are frozen temporary
        # constants outside the active student parameter/optimizer namespace.
        params=dict(student.named_parameters()); params['embedding_dict.user_emb']=self.initial_deleted.index_copy(0,self.keep,student.embedding_dict['user_emb'])
        return functional_call(self.original,params,(mode,))
    def __call__(self,student,u,p,n,indices):
        fu,fi=student(); av,_=student('adaptive'); bv,_=student()
        rec=F.softplus(-(fu[u]*(fi[p]-fi[n])).sum(1))
        cl=F.cross_entropy(F.normalize(av[u],dim=1)@F.normalize(bv[u],dim=1).T/student.ssl_temp,torch.arange(len(u),device=u.device),reduction='none')
        rw=torch.ones_like(rec) if self.uniform else self.weights['rec'][indices].detach(); cw=torch.ones_like(cl) if self.uniform or not self.contrastive_weighting else self.weights['anchor'][u].detach()
        retain_rec=(rw*rec).mean(); retain_cl=(cw*cl).mean()
        ids=self.rng.integers(len(self.forget_pairs),size=min(self.forget_batch,len(self.forget_pairs))); pairs=self.forget_pairs[ids]
        du,dp=torch.as_tensor(pairs,device=u.device).T; dn=torch.as_tensor(self.sample_negatives(pairs[:,0]),device=u.device)
        # Restore backbone dropout for request-side optimization; its parameters
        # stay frozen, while functional substitutes carry student gradients.
        self.original.train(student.training)
        old_u,old_i=self.request_forward(student); old_av,_=self.request_forward(student,'adaptive'); old_bv,_=self.request_forward(student)
        fw=torch.ones(len(du),device=u.device) if self.uniform else self.forget_rec[torch.as_tensor(ids,device=u.device)]
        aw=torch.ones(len(du),device=u.device) if self.uniform or not self.contrastive_weighting else self.forget_anchor[du]
        forget_rec=(fw*F.softplus((old_u[du]*(old_i[dp]-old_i[dn])).sum(1))).mean()
        forget_cl=weighted_infonce(old_av[du],old_bv[du],old_bv[self.keep[u.unique()]],student.ssl_temp,aw)
        retain=retain_rec+student.ssl_reg*retain_cl
        forget=forget_rec-self.forget_cl_strength*student.ssl_reg*forget_cl
        reg=student.reg*(fu[u].norm()+fi[p].norm()+fi[n].norm())
        # Reverse the native BPR pair ordering instead of unbounded BPR ascent.
        # Normalized InfoNCE forgetting ascent is bounded by its cosine logits.
        # Regularization always retains its minimizing sign.
        loss=retain+self.forget_strength*forget+reg
        self.components=dict(retain=retain+reg,forget=forget)
        self.last_components={k:float(v.detach()) for k,v in dict(retain_rec=retain_rec,retain_cl=retain_cl,forget_rec=forget_rec,forget_cl=forget_cl,total=loss).items()}
        return loss
