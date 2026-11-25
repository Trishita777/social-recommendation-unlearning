"""Shared ODSR data, training loss, and ranking utilities for the partition benchmark."""
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'external/ODSR'), str(ROOT/'src')]
from ODSR import ODSR
from common import seed_everything, _safe_read_pairs, _sp_to_tensor


def digest(x):
    return hashlib.sha256(np.asarray(x, dtype=np.int64).tobytes()).hexdigest()


def atomic_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(obj, indent=2, allow_nan=False)); os.replace(tmp, path)


def load_data(dataset, seed):
    if dataset == 'recflow-yelp':
        import zipfile
        import pickle
        path = ROOT/'external/RecFlowData/yelp.zip'
        if hashlib.sha256(path.read_bytes()).hexdigest() != 'c5e9155f25da2b707f1ceac37588da22ca7e6419efa16fa8daef0eac0d453633':
            raise ValueError('Unexpected RecFlow Yelp archive')
        with zipfile.ZipFile(path) as bundle:
            native = pickle.loads(bundle.read('dataset.pkl'))
        def pairs(key):
            return np.column_stack(native[key].tocsr().nonzero()).astype(np.int64)
        train, val, test, social = (pairs(key) for key in ('train','val','test','trust'))
        meta = dict(split_hash=digest(np.concatenate([train,[[-1,-1]],val,[[-2,-2]],test])), social_hash=digest(social))
        return SimpleNamespace(train=train,val=val,test=test,social=social,U=native['userCount'],I=native['itemCount'],meta=meta)
    base = ROOT/'external/ODSR/dataset'/dataset
    files = [base/'train.txt', base/'test.txt', base/('trust.txt' if (base/'trust.txt').exists() else 'links.txt')]
    for p in files:
        if not p.exists() or p.read_bytes().startswith(b'version https://git-lfs.github.com/spec/v1'):
            raise ValueError(f'Missing data or unresolved LFS: {p}')
    train_raw, test_raw, social_raw = [_safe_read_pairs(str(p)) for p in files]
    train_raw = list(dict.fromkeys(train_raw)); test_raw = list(dict.fromkeys(test_raw))
    overlap = set(train_raw)&set(test_raw)
    # Preserve supplied test; remove overlapping pairs from the training source.
    train_raw = [x for x in train_raw if x not in overlap]
    um = {u:k for k,u in enumerate(dict.fromkeys(u for u,i in train_raw))}
    im = {i:k for k,i in enumerate(dict.fromkeys(i for u,i in train_raw))}
    by = defaultdict(list)
    for u,i in train_raw: by[um[u]].append(im[i])
    rng = np.random.default_rng(seed); train=[]; val=[]
    for u, items in by.items():
        items=np.array(sorted(items)); rng.shuffle(items)
        n=min(len(items)-1, max(1, int(len(items)*.1))) if len(items)>1 else 0
        val.extend((u,int(i)) for i in items[:n]); train.extend((u,int(i)) for i in items[n:])
    test=[(um[u], im[i]) for u,i in test_raw if u in um and i in im]
    social=sorted(set((um[u],um[v]) for u,v in social_raw if u in um and v in um and u!=v))
    assert not set(train)&set(val) and not set(train)&set(test) and not set(val)&set(test)
    arr=lambda x:np.array(x,dtype=np.int64).reshape(-1,2)
    train,val,test,social=map(arr,(train,val,test,social))
    meta=dict(dataset=dataset,seed=seed,users=len(um),items=len(im),train=len(train),validation=len(val),test=len(test),raw_directed_social_edges=len(social),overlap_removed_from_train=len(overlap),test_unseen_excluded=len(test_raw)-len(test),social_only_users_excluded=len(set(x for p in social_raw for x in p)-set(um)),source_files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},split_hash=digest(np.concatenate([train, np.array([[-1,-1]]),val,np.array([[-2,-2]]),test])),social_hash=digest(social),social_policy='ODSR: symmetrize, deduplicate, add retained-user self loops, row normalize')
    return SimpleNamespace(train=train,val=val,test=test,social=social,U=len(um),I=len(im),um=um,im=im,meta=meta)


def graph(raw, keep, device):
    t0=time.perf_counter(); keep=np.asarray(keep); mapping=np.full(raw.U,-1,dtype=np.int64); mapping[keep]=np.arange(len(keep))
    mask=mapping[raw.train[:,0]]>=0; train=raw.train[mask].copy(); train[:,0]=mapping[train[:,0]]
    smask=(mapping[raw.social[:,0]]>=0)&(mapping[raw.social[:,1]]>=0)
    social=mapping[raw.social[smask]]
    U=len(keep); I=raw.I
    ui=sp.csr_matrix((np.ones(len(train),np.float32),(train[:,0],train[:,1])),shape=(U,I))
    adj=sp.bmat([[None,ui],[ui.T,None]],format='csr'); deg=np.asarray(adj.sum(1)).ravel(); inv=np.zeros_like(deg); np.power(deg,-.5,out=inv,where=deg>0)
    norm=sp.diags(inv)@adj@sp.diags(inv)
    soc=sp.csr_matrix((np.ones(len(social),np.float32),(social[:,0],social[:,1])),shape=(U,U)); soc=soc.maximum(soc.T); soc.setdiag(1); soc.eliminate_zeros()
    ds=np.asarray(soc.sum(1)).ravel(); uu=sp.diags(1/ds)@soc
    positive=[set() for _ in keep]
    for u,i in train: positive[u].add(int(i))
    def mapped(pairs):
        p=pairs[mapping[pairs[:,0]]>=0].copy(); p[:,0]=mapping[p[:,0]]; return p
    out=SimpleNamespace(user_num=U,item_num=I,adj_ui=_sp_to_tensor(ui,device),adj_uu=_sp_to_tensor(uu,device),norm_adj=_sp_to_tensor(norm,device),train=train,val=mapped(raw.val),test=mapped(raw.test),keep=keep,mapping=mapping,positives=positive,ui=ui,social=social)
    out.counts=dict(users=U,interactions=len(train),directed_social_edges=len(social),removed_users=raw.U-U,removed_interactions=len(raw.train)-len(train),removed_directed_social_edges=len(raw.social)-len(social),isolated_items=int((np.asarray(ui.sum(0)).ravel()==0).sum()),isolated_users=int((np.asarray(ui.sum(1)).ravel()==0).sum()),graph_seconds=time.perf_counter()-t0)
    return out


def negatives(data, users, rng):
    n=rng.integers(data.item_num,size=len(users))
    for k,u in enumerate(users):
        if len(data.positives[u])==data.item_num: raise ValueError('No negative item')
        while int(n[k]) in data.positives[u]: n[k]=rng.integers(data.item_num)
    return n


@torch.no_grad()
def evaluate(model,data,pairs,device):
    model.eval(); U,I=model(); truth=defaultdict(set)
    for u,i in pairs: truth[int(u)].add(int(i))
    totals=defaultdict(float); ranks={}; n=0
    for start in range(0,len(truth),128):
        users=list(truth)[start:start+128]; scores=U[users]@I.T
        for row,u in enumerate(users):
            if data.positives[u]: scores[row,list(data.positives[u])]=-torch.inf
        ids=scores.topk(min(20,data.item_num),dim=1).indices.cpu().numpy()
        for u,top in zip(users,ids):
            n+=1; hit=np.array([i in truth[u] for i in top]); row={}
            for k in (10,20):
                h=hit[:k]; denom=min(len(truth[u]),k)
                row[f'precision@{k}']=float(h.sum()/k); row[f'recall@{k}']=float(h.sum()/len(truth[u]))
                row[f'map@{k}']=float(((np.cumsum(h)/np.arange(1,len(h)+1))*h).sum()/denom)
                row[f'ndcg@{k}']=float((h/np.log2(np.arange(2,len(h)+2))).sum()/(1/np.log2(np.arange(2,denom+2))).sum())
            for key,value in row.items(): totals[key]+=value
            ranks[u]=top
    result={key:value/max(n,1) for key,value in totals.items()}; result['evaluated_users']=n
    return result, (U.detach(), I.detach(), ranks)


def loss_terms(model,u,p,n):
    fu,fi=model(); rec=F.softplus(-((fu[u]*(fi[p]-fi[n])).sum(1)))
    av,_=model('adaptive'); bv,_=model(); v1=F.normalize(av[u],dim=1); v2=F.normalize(bv[u],dim=1)
    # Preserve official ODSR's batch candidate pool, including repeated-user multiplicity.
    logits=v1@v2.T/model.ssl_temp; cl=F.cross_entropy(logits,torch.arange(len(u),device=u.device),reduction='none')
    return rec.mean() + model.ssl_reg * cl.mean() + model.reg * (fu[u].norm() + fi[p].norm() + fi[n].norm())
