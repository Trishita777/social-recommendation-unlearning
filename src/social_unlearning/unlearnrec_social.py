"""Execute the official UnlearnRec encoder with a PyTorch sparse adapter.
Backbone wrapper supplies joint social-user requests and native ODSR recovery.
Original encoder, withdrawal, alignment and SSL functions are source-executed.
"""
import ast
from types import SimpleNamespace
import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F
from torch.func import functional_call


class SparseCompat:
    def __init__(self,row=None,col=None,value=None,sparse_sizes=None,tensor=None):
        self.tensor=tensor.coalesce() if tensor is not None else torch.sparse_coo_tensor(torch.stack([row,col]),value,sparse_sizes,device=value.device).coalesce()
    def coo(self):return self.tensor.indices()[0],self.tensor.indices()[1],self.tensor.values()
    def nnz(self):return self.tensor._nnz()
    def sizes(self):return self.tensor.shape
    def matmul(self,x):return torch.sparse.mm(self.tensor,x)
    def cuda(self):return SparseCompat(tensor=self.tensor.cuda())


def encoder_graphs(raw,keep,device):
    U,I=raw.U,raw.I;train=raw.train;social=raw.social
    ui=sp.csr_matrix((np.ones(len(train),np.float32),(train[:,0],train[:,1])),shape=(U,I));uu=sp.csr_matrix((np.ones(len(social),np.float32),(social[:,0],social[:,1])),shape=(U,U));uu=uu.maximum(uu.T)
    full=sp.bmat([[uu,ui],[ui.T,None]],format='csr');mask=np.zeros(U+I,np.float32);mask[keep]=1;mask[U:]=1;retained=sp.diags(mask)@full@sp.diags(mask);dropped=full-retained
    def normalized(adj):
        degree=np.asarray(adj.sum(1)).ravel();inv=np.zeros_like(degree);np.power(degree,-.5,out=inv,where=degree>0);coo=(sp.diags(inv)@adj@sp.diags(inv)).tocoo();tensor=torch.sparse_coo_tensor(torch.tensor(np.vstack([coo.row,coo.col]),dtype=torch.long,device=device),torch.tensor(coo.data,device=device),coo.shape).coalesce();return SparseCompat(tensor=tensor)
    return tuple(normalized(x) for x in [full,retained,dropped])


class SocialBackbone(nn.Module):
    def __init__(self,original,original_data,cfg,device):
        super().__init__();self.original=original;self.original_data=original_data;self.cfg=cfg;self.device=device;self.U=original.user_num
        self.ini_embeds=nn.Parameter(torch.cat([original.embedding_dict['user_emb'],original.embedding_dict['item_emb']]).detach().clone(),requires_grad=False)
        for p in original.parameters():p.requires_grad_(False)
    def set_request(self,data):
        from ODSR import ODSR
        self.data=data;self.keep=torch.as_tensor(data.keep,device=self.device);self.mapping=torch.as_tensor(data.mapping,device=self.device);self.body=ODSR(data,self.cfg,self.device).to(self.device)
        parameters={k:v.detach().clone() for k,v in self.original.named_parameters()};parameters['embedding_dict.user_emb']=parameters['embedding_dict.user_emb'][self.keep];self.body.load_state_dict(parameters)
        for p in self.body.parameters():p.requires_grad_(False)
    def native(self,tuned,mode=None):
        return functional_call(self.body,{'embedding_dict.user_emb':tuned[:self.U][self.keep],'embedding_dict.item_emb':tuned[self.U:]},(mode,))
    def forward(self,adj,tuned=None,keepRate=None):
        tuned=self.ini_embeds if tuned is None else tuned;old_body=self.body.training;old_original=self.original.training
        if keepRate==1.:self.body.eval();self.original.eval()
        u,i=self.native(tuned);old_u,_=functional_call(self.original,{'embedding_dict.user_emb':tuned[:self.U],'embedding_dict.item_emb':tuned[self.U:]},())
        full_u=old_u.index_copy(0,self.keep,u)
        if keepRate==1.:self.body.train(old_body);self.original.train(old_original)
        return full_u,i
    def cal_loss(self,batch_data,tuned_emb=None,**kwargs):
        global_u,p,n=batch_data;u=self.mapping[global_u]
        if (u<0).any():raise ValueError('Deleted anchor entered retained recovery loss')
        fu,fi=self.native(tuned_emb);a,_=self.native(tuned_emb,'adaptive');b,_=self.native(tuned_emb)
        rec=F.softplus(-(fu[u]*(fi[p]-fi[n])).sum(1)).mean();cl=F.cross_entropy(F.normalize(a[u],dim=1)@F.normalize(b[u],dim=1).T/self.body.ssl_temp,torch.arange(len(u),device=self.device));reg=self.body.reg*(fu[u].norm()+fi[p].norm()+fi[n].norm());return rec+self.body.ssl_reg*cl+reg,dict(bpr_loss=rec,reg_loss=reg,backbone_ssl=cl)
    @torch.no_grad()
    def folded(self,tuned):
        from ODSR import ODSR
        model=ODSR(self.data,self.cfg,self.device).to(self.device);parameters={k:v.detach().clone() for k,v in self.body.named_parameters()};parameters['embedding_dict.user_emb']=tuned[:self.U][self.keep].detach().clone();parameters['embedding_dict.item_emb']=tuned[self.U:].detach().clone();model.load_state_dict(parameters);return model


def load_official_encoder(source_root,args,wrapper,initial,final,handler):
    """Execute selected classes/functions unchanged; avoid unrelated imports."""
    namespace=dict(t=torch,torch=torch,nn=nn,F=F,np=np,args=args,ts=SimpleNamespace(SparseTensor=SparseCompat))
    utils=ast.parse((source_root/'Utils/utils.py').read_text());functions={'cal_neg_aug_v1','cal_neg_aug_v2','cal_l2_distance','innerProduct','cal_positive_pred_align_v2','cal_positive_pred_align_v3','contrastLoss'}
    selected=[n for n in utils.body if isinstance(n,ast.FunctionDef) and n.name in functions];exec(compile(ast.fix_missing_locations(ast.Module(body=selected,type_ignores=[])),str(source_root/'Utils/utils.py'),'exec'),namespace)
    tree=ast.parse((source_root/'Model.py').read_text());classes={'SpAdjDropEdge','GCNLayer','GraphUnlearning','FeedForwardLayer'};selected=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in classes];exec(compile(ast.fix_missing_locations(ast.Module(body=selected,type_ignores=[])),str(source_root/'Model.py'),'exec'),namespace)
    namespace['cal_positive_pred_align_v2']=stable_positive_alignment
    return namespace['GraphUnlearning'](handler,wrapper,initial,final)


def stable_positive_alignment(srcUEmbs,tarUEmbs,srcIEmbs,tarIEmbs,align,temp=1.):
    """Same official v2 expression/epsilons, evaluated in log space to avoid exp overflow."""
    def distribution(u,i):
        eps=u.new_tensor(1e-6).log()
        positive=(u*i).sum(-1)/temp
        denominator=torch.logaddexp(torch.logsumexp(u@i.T/temp+1e-8,dim=-1),eps)
        return torch.logaddexp(positive-denominator,eps)
    return align(distribution(srcUEmbs,srcIEmbs),distribution(tarUEmbs,tarIEmbs))
