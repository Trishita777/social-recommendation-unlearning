"""Paper-derived RRL social adapter (AAAI 2024, Eq. 2--3).

Retains the log-sigmoid reverse objective, NOT reverse-softplus ranking.
The empirical Fisher is a sampled low-rank outer-product matrix, not a
diagonal approximation: cross-parameter terms survive. Original-graph
forwards are request-local witnesses; deployment uses rebuilt graphs.
"""
import numpy as np
import torch
from torch.func import functional_call
from torch.nn import functional as F


def empirical_fisher_penalty(parameters, initial, factors):
    """delta.T E[g g.T] delta without materializing the dense matrix."""
    if not factors:
        raise ValueError('Fisher requires at least one gradient sample')
    return torch.stack([sum(((p-q)*g).sum() for p,q,g in zip(parameters,initial,row)).square()
                        for row in factors]).mean()


class RRL:
    def __init__(self, original, student, raw, deleted, names, seed, fisher_samples=64):
        self.original=original;self.student=student;self.names=names
        self.rng=np.random.default_rng(seed);self.raw=raw
        self.pairs=raw.train[np.isin(raw.train[:,0],deleted)]
        if not len(self.pairs):raise ValueError('No requested interactions')
        self.positive=[set() for _ in range(raw.U)]
        for u,i in raw.train:self.positive[u].add(int(i))
        self.params=[dict(student.named_parameters())[n] for n in names]
        self.initial=[dict(original.named_parameters())[n].detach().clone() for n in names]
        for n,p in student.named_parameters():p.requires_grad_(n in names)
        for p in original.parameters():p.requires_grad_(False)
        original.eval();student.eval()
        # Each factor is a per-triplet BPR gradient at the original parameters.
        # Fixed uniform training probes and true unseen negatives are recorded.
        self.probe_ids=self.rng.choice(len(raw.train),min(fisher_samples,len(raw.train)),replace=False)
        pairs=raw.train[self.probe_ids];negative=self.negatives(pairs[:,0])
        self.probe_negatives=negative.copy();self.factors=[]
        u,i=self.witness()
        for k,((a,b),c) in enumerate(zip(pairs,negative)):
            loss=F.softplus(-(u[a]*(i[b]-i[c])).sum())
            gs=torch.autograd.grad(loss,self.params,retain_graph=k+1<len(pairs))
            self.factors.append(tuple(g.detach() for g in gs))
        self.last_components={}

    def negatives(self,users):
        result=self.rng.integers(self.raw.I,size=len(users))
        for k,u in enumerate(users):
            if len(self.positive[u])==self.raw.I:raise ValueError('No unseen negative')
            while int(result[k]) in self.positive[u]:result[k]=self.rng.integers(self.raw.I)
        return result

    def witness(self):
        # Student tensors carry gradients, original graph buffers remain local.
        return functional_call(self.original,dict(self.student.named_parameters()),())

    def loss(self,pairs):
        a,b=torch.as_tensor(pairs,device=self.params[0].device).T
        c=torch.as_tensor(self.negatives(pairs[:,0]),device=a.device)
        u,i=self.witness();margin=(u[a]*(i[b]-i[c])).sum(1)
        reverse=F.logsigmoid(margin).sum()
        preservation=empirical_fisher_penalty(self.params,self.initial,self.factors)
        loss=reverse+preservation
        self.last_components=dict(reverse=float(reverse.detach()),fisher=float(preservation.detach()),total=float(loss.detach()))
        return loss
