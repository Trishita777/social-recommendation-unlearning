"""Equation port of external/EGCL/model/social_recommender/EGCL.py.
Status: numerical unit checks only; TensorFlow cross-framework parity and
full-data convergence validation are required before paper-quality results.
No silent replacement of EGCL's full-batch objective with a BPR surrogate.
"""
import math
import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def sparse(matrix,device):
    coo=matrix.tocoo(); return torch.sparse_coo_tensor(torch.tensor(np.vstack([coo.row,coo.col]),dtype=torch.long),torch.tensor(coo.data,dtype=torch.float32),coo.shape,device=device).coalesce()


def symmetric_normalize(adj):
    deg=np.asarray(adj.sum(1)).ravel(); d=np.zeros_like(deg); np.power(deg,-.5,out=d,where=deg>0)
    return sp.diags(d)@adj@sp.diags(d),deg


class Attention(nn.Module):
    def __init__(self,dim):
        super().__init__(); self.W=nn.Parameter(torch.empty(2*dim,dim)); self.b=nn.Parameter(torch.empty(1,dim)); self.h=nn.Parameter(torch.ones(dim,1))
        nn.init.trunc_normal_(self.W,std=.01,a=-.02,b=.02); nn.init.trunc_normal_(self.b,std=.01,a=-.02,b=.02)
    def forward(self,layers,initial):
        cat=torch.cat([layers,initial[:,None,:].expand_as(layers)],dim=2)
        scores=(F.relu(cat@self.W+self.b)@self.h).squeeze(-1)
        return (scores.softmax(1)[:,:,None]*layers).sum(1)


class EGCL(nn.Module):
    def __init__(self,data,cfg,device):
        super().__init__(); self.user_num=data.user_num; self.item_num=data.item_num; self.cfg=cfg; self.device=device
        d=cfg.get('embedding_size',64); self.layers=cfg.get('layer_size',3)
        self.user_embeddings=nn.Parameter(torch.empty(self.user_num,d)); self.item_embeddings=nn.Parameter(torch.empty(self.item_num,d)); nn.init.xavier_uniform_(self.user_embeddings); nn.init.xavier_uniform_(self.item_embeddings)
        self.user_attention=Attention(d); self.item_attention=Attention(d); self.social_attention=Attention(d)
        self.rec_weight=nn.Sequential(nn.Linear(3*d,d),nn.ReLU(),nn.Linear(d,1)); self.social_weight=nn.Sequential(nn.Linear(3*d,d),nn.ReLU(),nn.Linear(d,1))
        for module in (self.rec_weight,self.social_weight):
            for p in module.parameters():nn.init.trunc_normal_(p,std=.01,a=-.02,b=.02)
        ui=data.ui.astype(np.float32).tocsr(); ui.data[:]=1
        social=data.social
        uu=sp.csr_matrix((np.ones(len(social),np.float32),(social[:,0],social[:,1])),shape=(self.user_num,self.user_num))
        # Official EGCL adds directed adjacency and its transpose. Reciprocal edges
        # therefore have weight two; no self loops are added.
        recadj=sp.bmat([[None,ui],[ui.T,None]],format='csr'); socadj=uu+uu.T
        norm,degree=symmetric_normalize(recadj); snorm,sdegree=symmetric_normalize(socadj)
        self.register_buffer('recadj',sparse(norm,device)); self.register_buffer('socadj',sparse(snorm,device))
        # sigmoid(log(degree + 1e-7)), as in TensorFlow source.
        self.register_buffer('degree_time',torch.sigmoid(torch.log(torch.tensor(degree,device=device)+1e-7))[:,None]); self.register_buffer('social_time',torch.sigmoid(torch.log(torch.tensor(sdegree,device=device)+1e-7))[:,None])
        def column_normalize(a):
            degree=np.asarray(a.sum(0)).ravel(); inv=np.zeros_like(degree); np.divide(1,degree,out=inv,where=degree>0); return a@sp.diags(inv)
        self.register_buffer('rating_neighbors',sparse(column_normalize(ui),device)); self.register_buffer('social_neighbors',sparse(column_normalize(uu),device))
        self.register_buffer('interaction_index',torch.as_tensor(np.column_stack(ui.nonzero()),device=device)); self.register_buffer('social_index',torch.as_tensor(social,device=device))
        self.to(device)
    def propagate(self,initial,adj,t):
        agg=initial; layers=[]
        for k in range(self.layers):
            agg=torch.sparse.mm(adj,agg); layers.append(agg*torch.exp(-t)*t.pow(k)/math.factorial(k))
        return torch.stack(layers,dim=1)
    def views(self):
        initial=torch.cat([self.user_embeddings,self.item_embeddings]); layers=self.propagate(initial,self.recadj,self.degree_time)
        u=self.user_attention(layers[:self.user_num],self.user_embeddings); i=self.item_attention(layers[self.user_num:],self.item_embeddings)
        social=self.social_attention(self.propagate(self.user_embeddings,self.socadj,self.social_time),self.user_embeddings)
        return u,i,social
    def forward(self):
        u,i,_=self.views(); return u,i
    @staticmethod
    def fast_loss(a,b,index,alpha):
        # Exact Gram identity for all-pairs squared predictions, including nonedges.
        allpairs=((a.T@a)*(b.T@b)).sum(); pos=(a[index[:,0]]*b[index[:,1]]).sum(1)
        return allpairs+((alpha-1)*pos.square()-2*alpha*pos).sum(),alpha.sum()
    @staticmethod
    def l2(parameters):return sum(p.square().sum()/2 for p in parameters)
    def info_nce(self,a,b,anchor_weights=None):
        a=F.normalize(a,dim=1,eps=1e-6); b=F.normalize(b,dim=1,eps=1e-6); loss=a.new_zeros(()); size=self.cfg.get('contrastive_chunk',512); temp=self.cfg.get('temperature',.05)
        # Full candidate denominator; checkpoint chunks to avoid quadratic saved activations.
        for start in range(0,len(a),size):
            end=min(start+size,len(a)); labels=torch.arange(start,end,device=a.device)
            def chunk(x,y,labels):return F.cross_entropy(x@y.T/temp,labels,reduction='none')
            values=checkpoint(chunk,a[start:end],b,labels,use_reentrant=False) if self.training and torch.is_grad_enabled() else chunk(a[start:end],b,labels)
            loss=loss+(values if anchor_weights is None else values*anchor_weights[start:end].detach()).sum()
        return loss
    def embedding_parameters(self):
        return [self.user_embeddings,self.item_embeddings,*self.user_attention.parameters(),*self.item_attention.parameters(),*self.social_attention.parameters()]
    def weight_parameters(self):return list(self.rec_weight.parameters())+list(self.social_weight.parameters())
    def objectives(self,include_cl=True,anchor_weights=None):
        cfg=self.cfg; u,i,s=self.views(); ui=self.interaction_index; uu=self.social_index
        def alpha(a,b,index,network,offset):
            x=a[index[:,0]]; y=b[index[:,1]]; return F.softplus(network(torch.cat([x,y,x*y],1)).squeeze(1))+offset
        rec,rec3=self.fast_loss(u,i,ui,alpha(u,i,ui,self.rec_weight,8))
        soc,soc3=self.fast_loss(s,s,uu,alpha(s,s,uu,self.social_weight,10))
        wr=cfg.get('weight_reg',60); beta=cfg.get('beta',.3)
        weight_loss=rec+rec3+wr*self.l2(self.rec_weight.parameters())+beta*(soc+soc3+wr*self.l2(self.social_weight.parameters()))
        embedding=rec+cfg.get('embedding_reg',2)*self.l2([self.user_embeddings,self.item_embeddings])+wr*self.l2([*self.user_attention.parameters(),*self.item_attention.parameters()])+beta*(soc+wr*self.l2(self.social_attention.parameters()))
        if not include_cl:return weight_loss,embedding
        weights=anchor_weights or {}
        intra=self.info_nce(u,torch.sparse.mm(self.rating_neighbors,i),weights.get('user'))+self.info_nce(s,torch.sparse.mm(self.social_neighbors,s),weights.get('social'))+self.info_nce(i,torch.sparse.mm(self.rating_neighbors.transpose(0,1),u),weights.get('item'))
        inter=self.info_nce(u,s,weights.get('inter_user'))
        return weight_loss,embedding+cfg.get('intra_reg',1e-6)*intra+cfg.get('inter_reg',1e-8)*inter
    def alternating_step(self,weight_optimizer,embedding_optimizer):
        self.train(); weight_optimizer.zero_grad(set_to_none=True); wl,_=self.objectives(include_cl=False); params=self.weight_parameters(); grads=torch.autograd.grad(wl,params)
        for p,g in zip(params,grads):p.grad=g
        weight_optimizer.step(); embedding_optimizer.zero_grad(set_to_none=True); _,el=self.objectives(); params=self.embedding_parameters(); grads=torch.autograd.grad(el,params)
        for p,g in zip(params,grads):p.grad=g
        embedding_optimizer.step(); return float(wl.detach()),float(el.detach())
