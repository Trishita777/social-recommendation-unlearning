"""Joint incident-edge deletion with persistent user IDs and embedding rows."""
import copy
from types import SimpleNamespace
import numpy as np
import torch
from torch.nn import functional as F
from torch.func import functional_call
from .native_guarded import normalized_weights
from .native_guarded_memory import MemoryGuardedNativeForgetRetain


def filtered_raw(raw, deleted):
    deleted = np.asarray(deleted, dtype=np.int64)
    if len(np.unique(deleted)) != len(deleted) or np.any(deleted < 0) or np.any(deleted >= raw.U):
        raise ValueError('Invalid requested user IDs')
    result = copy.copy(raw)
    result.train = raw.train[~np.isin(raw.train[:,0], deleted)].copy()
    result.social = raw.social[~np.isin(raw.social, deleted).any(1)].copy()
    # Utility is retained-user utility; attack histories are separate audit data.
    result.val = raw.val[~np.isin(raw.val[:,0], deleted)].copy()
    result.test = raw.test[~np.isin(raw.test[:,0], deleted)].copy()
    return result


def edge_graph(raw, deleted, device):
    from run_unlearning import graph
    filtered = filtered_raw(raw, deleted)
    data = graph(filtered, np.arange(raw.U), device)
    data.deleted_users = np.asarray(deleted, dtype=np.int64)
    data.retained_users = np.setdiff1d(np.arange(raw.U), data.deleted_users)
    # ODSR adds synthetic self loops; deleted users must be fully isolated.
    adjacency = data.adj_uu.coalesce()
    mask = ~torch.isin(adjacency.indices(), torch.as_tensor(deleted, device=device)).any(0)
    data.adj_uu = torch.sparse_coo_tensor(adjacency.indices()[:,mask], adjacency.values()[mask], adjacency.shape, device=device).coalesce()
    data.counts.update(removed_users=0, requested_users=len(deleted),
        removed_interactions=len(raw.train)-len(data.train),
        removed_directed_social_edges=len(raw.social)-len(data.social),
        deletion_semantics='incident_edges_only_keep_user_rows')
    return data


def retained_proxy(data):
    # Only constructors need the active-user set to identify request histories.
    # Forward tensors and user IDs remain in the full, original row space.
    proxy = copy.copy(data)
    proxy.keep = data.retained_users
    return proxy


@torch.no_grad()
def edge_sensitivity(original, student, data, seed, mix=.25):
    from run_unlearning import negatives
    original.eval(); student.eval()
    ou,oi=original(); ru,ri=student()
    rng=np.random.default_rng(seed)
    neg=negatives(data,data.train[:,0],rng)
    changes=[]
    for start in range(0,len(data.train),4096):
        u,p=torch.as_tensor(data.train[start:start+4096],device=student.device).T
        n=torch.as_tensor(neg[start:start+4096],device=student.device)
        changes.append(((ou[u]*(oi[p]-oi[n])).sum(1)-(ru[u]*(ri[p]-ri[n])).sum(1)).abs())
    candidates=torch.as_tensor(rng.choice(data.retained_users,min(512,len(data.retained_users)),replace=False),device=student.device)
    deleted=torch.as_tensor(rng.choice(data.deleted_users,min(512,len(data.deleted_users)),replace=False),device=student.device)
    oc=F.normalize(ou[candidates],dim=1); rc=F.normalize(ru[candidates],dim=1)
    dc=F.normalize(ou[deleted],dim=1); scores=[]
    for start in range(0,data.user_num,512):
        a=F.normalize(ou[start:start+512],dim=1); b=F.normalize(ru[start:start+512],dim=1)
        lo=a@oc.T/student.ssl_temp; ln=b@rc.T/student.ssl_temp
        po,pn=lo.softmax(1),ln.softmax(1); mean=(po+pn)/2
        js=.5*((po*(po.clamp_min(1e-12).log()-mean.clamp_min(1e-12).log())).sum(1)+(pn*(pn.clamp_min(1e-12).log()-mean.clamp_min(1e-12).log())).sum(1))
        mass=torch.cat([lo,a@dc.T/student.ssl_temp],1).softmax(1)[:,lo.shape[1]:].sum(1)
        scores.append(js+mass)
    rec=normalized_weights(torch.cat(changes))
    score=torch.cat(scores)
    active=torch.as_tensor(data.retained_users,device=student.device)
    anchor=torch.ones(data.user_num,device=student.device)
    anchor[active]=normalized_weights(score[active])
    return dict(rec=1+mix*(rec-1),anchor=1+mix*(anchor-1))


class EdgeGuardedForgetRetain(MemoryGuardedNativeForgetRetain):
    """Same staged guard and native loss forms; requested rows remain trainable.

    Both retention and forgetting forwards use the rebuilt edge-deleted graph.
    The original model supplies detached sensitivity estimates only. Retention batches/candidates
    contain active users. This is a new experimental variant, not old V4 results.
    """
    def __init__(self, original, student, data, raw, weights, seed, **kwargs):
        super().__init__(original,student,retained_proxy(data),raw,weights,seed,**kwargs)
        self.keep=torch.arange(raw.U,device=student.device)
        self.data=data

    def request_forward(self, student, mode=None):
        return student(mode)


def edge_da2(original,student,data,raw,original_data,source,seed,cfg):
    from .baselines import DA2Social
    objective=DA2Social(original,student,retained_proxy(data),raw,original_data,source,seed,cfg)
    objective.request.keep=torch.arange(raw.U,device=student.device)
    # Joint-edge adaptation: all optimized predictions use altered adjacencies.
    objective.request.forward=lambda model,mode=None:model(mode)
    return objective
