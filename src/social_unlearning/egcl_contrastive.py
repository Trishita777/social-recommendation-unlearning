"""Fixed-ID sample-weighted contrastive forgetting/retention for EGCL.

Native all-pairs reconstruction, social loss, regularization, four full-pool
contrastive terms and alternating parameter blocks are retained. Requested
users are excluded from retained CL pools. Forgetting requires a temporary
original-graph witness: isolated EGCL users have exactly zero deployed views.
"""
import numpy as np
import torch
from torch import nn
from torch.func import functional_call
from torch.nn import functional as F
from .native_guarded import normalized_weights
from .contrastive_core import forget_infonce


def native_retention(model, rec_weights=None, anchor_weights=None, retained=None, include_cl=True):
    cfg=model.cfg;u,i,s=model.views();ui=model.interaction_index;uu=model.social_index
    def alpha(a,b,index,network,offset):
        x,y=a[index[:,0]],b[index[:,1]]
        return F.softplus(network(torch.cat([x,y,x*y],1)).squeeze(1))+offset
    ar=alpha(u,i,ui,model.rec_weight,8)
    if rec_weights is not None:ar=ar*rec_weights.detach()
    rec,rec3=model.fast_loss(u,i,ui,ar)
    soc,soc3=model.fast_loss(s,s,uu,alpha(s,s,uu,model.social_weight,10))
    wr=cfg.get('weight_reg',60);beta=cfg.get('beta',.3)
    weight=rec+rec3+wr*model.l2(model.rec_weight.parameters())+beta*(soc+soc3+wr*model.l2(model.social_weight.parameters()))
    embedding=rec+cfg.get('embedding_reg',2)*model.l2([model.user_embeddings,model.item_embeddings])
    embedding+=wr*model.l2([*model.user_attention.parameters(),*model.item_attention.parameters()])+beta*(soc+wr*model.l2(model.social_attention.parameters()))
    if not include_cl:return weight,embedding
    weights=anchor_weights or {}
    uv=torch.sparse.mm(model.rating_neighbors,i);sv=torch.sparse.mm(model.social_neighbors,s)
    iv=torch.sparse.mm(model.rating_neighbors.transpose(0,1),u)
    def cl(a,b,key):
        w=weights.get(key)
        if retained is not None and key!='item':
            a,b=a[retained],b[retained]
            if w is not None:w=w[retained]
        return model.info_nce(a,b,w)
    intra=cl(u,uv,'user')+cl(s,sv,'social')+cl(i,iv,'item')
    inter=cl(u,s,'inter_user')
    return weight,embedding+cfg.get('intra_reg',1e-6)*intra+cfg.get('inter_reg',1e-8)*inter


class _ViewWitness(nn.Module):
    def __init__(self,model):super().__init__();self.model=model
    def forward(self):return self.model.views()


class ContrastiveEGCL:
    def __init__(self,original,student,raw,data,seed,mix=.25):
        self.original=original;self.student=student;self.raw=raw;self.data=data
        self.rng=np.random.default_rng(seed);self.mix=mix;device=student.device
        self.deleted=torch.as_tensor(data.deleted_users,device=device)
        self.retained=torch.as_tensor(data.retained_users,device=device)
        self.pairs=raw.train[np.isin(raw.train[:,0],data.deleted_users)]
        self.positive=[set() for _ in range(raw.U)]
        for u,i in raw.train:self.positive[u].add(int(i))
        self.wrapper=_ViewWitness(original)
        for p in original.parameters():p.requires_grad_(False)
        self.last_components={};self.prepare_weights()

    def negatives(self,users):
        values=self.rng.integers(self.raw.I,size=len(users))
        for k,u in enumerate(users):
            if len(self.positive[u])==self.raw.I:raise ValueError('No negative items')
            while int(values[k]) in self.positive[u]:values[k]=self.rng.integers(self.raw.I)
        return values

    def mix_weights(self,values):return 1+self.mix*(normalized_weights(values)-1)

    @torch.no_grad()
    def prepare_weights(self):
        self.original.eval();self.student.eval();ou,oi,os=self.original.views();ru,ri,rs=self.student.views()
        def rec_sensitivity(pairs):
            neg=self.negatives(pairs[:,0]);values=[]
            for start in range(0,len(pairs),4096):
                a,b=torch.as_tensor(pairs[start:start+4096],device=ou.device).T;c=torch.as_tensor(neg[start:start+4096],device=ou.device)
                old=(ou[a]*(oi[b]-oi[c])).sum(1);new=(ru[a]*(ri[b]-ri[c])).sum(1);values.append((old-new).abs())
            return self.mix_weights(torch.cat(values))
        self.rec_weights=rec_sensitivity(self.data.train);self.forget_rec=rec_sensitivity(self.pairs)
        self.anchor_weights={};self.forget_anchor={}
        # Compare each native paired-view candidate distribution separately.
        oldpairs={'user':(ou,torch.sparse.mm(self.original.rating_neighbors,oi)),
                  'social':(os,torch.sparse.mm(self.original.social_neighbors,os)),
                  'item':(oi,torch.sparse.mm(self.original.rating_neighbors.transpose(0,1),ou)),
                  'inter_user':(ou,os)}
        newpairs={'user':(ru,torch.sparse.mm(self.student.rating_neighbors,ri)),
                  'social':(rs,torch.sparse.mm(self.student.social_neighbors,rs)),
                  'item':(ri,torch.sparse.mm(self.student.rating_neighbors.transpose(0,1),ru)),
                  'inter_user':(ru,rs)}
        temp=self.student.cfg.get('temperature',.05)
        for key,(a,b) in oldpairs.items():
            c,d=newpairs[key];pool=torch.arange(len(a),device=a.device) if key=='item' else self.retained
            ids=pool[torch.as_tensor(self.rng.choice(len(pool),min(512,len(pool)),replace=False),device=a.device)]
            deleted=self.deleted[:512] if key!='item' else None;changes=[]
            for start in range(0,len(a),512):
                old=F.normalize(a[start:start+512],dim=1)@F.normalize(b[ids],dim=1).T/temp
                new=F.normalize(c[start:start+512],dim=1)@F.normalize(d[ids],dim=1).T/temp
                po,pn=old.softmax(1),new.softmax(1);mean=(po+pn)/2
                js=.5*((po*(po.clamp_min(1e-12).log()-mean.clamp_min(1e-12).log())).sum(1)+(pn*(pn.clamp_min(1e-12).log()-mean.clamp_min(1e-12).log())).sum(1))
                if deleted is not None:
                    extra=F.normalize(a[start:start+512],dim=1)@F.normalize(b[deleted],dim=1).T/temp
                    js+=torch.cat([old,extra],1).softmax(1)[:,len(ids):].sum(1)
                changes.append(js)
            values=torch.cat(changes);w=torch.ones(len(a),device=a.device);w[pool]=self.mix_weights(values[pool]);self.anchor_weights[key]=w
            if key!='item':
                fw=torch.ones(len(a),device=a.device);fw[self.deleted]=self.mix_weights(values[self.deleted]);self.forget_anchor[key]=fw

    def witness(self):
        return functional_call(self.wrapper,{'model.'+n:p for n,p in self.student.named_parameters()},())

    def forgetting(self):
        ids=self.rng.choice(len(self.pairs),min(256,len(self.pairs)),replace=False)
        pairs=self.pairs[ids];a,b=torch.as_tensor(pairs,device=self.student.device).T
        c=torch.as_tensor(self.negatives(pairs[:,0]),device=a.device)
        u,i,s=self.witness();margin=(u[a]*(i[b]-i[c])).sum(1)
        rec=(self.forget_rec[torch.as_tensor(ids,device=a.device)]*F.softplus(margin)).mean()
        # All retained users are candidates, not an optimization-time probe.
        uv=torch.sparse.mm(self.original.rating_neighbors,i)
        sv=torch.sparse.mm(self.original.social_neighbors,s);temp=self.student.cfg.get('temperature',.05)
        def term(x,y,key):return (forget_infonce(x[a],y[a],y[self.retained],temp)*self.forget_anchor[key][a].detach()).mean()
        intra=term(u,uv,'user')+term(s,sv,'social');inter=term(u,s,'inter_user')
        cl=self.student.cfg.get('intra_reg',1e-6)*intra+self.student.cfg.get('inter_reg',1e-8)*inter
        forget=rec-.1*cl
        self.last_components=dict(forget_rec=float(rec.detach()),forget_cl=float(cl.detach()))
        return forget

    def retention(self,include_cl=True):
        return native_retention(self.student,self.rec_weights,self.anchor_weights,self.retained,include_cl)
