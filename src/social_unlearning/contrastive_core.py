"""Backbone-independent contrastive unlearning loss primitives.
Adapters supply scores, paired views, native retained losses, graph rebuilding,
request-side parameter substitution and optimizer blocks. This module supplies
no ODSR/EGCL adapter and makes no empirical or exact-erasure claim.
"""
import torch
from torch.nn import functional as F


def weighted_mean(values,weights):
    if values.shape!=weights.shape or values.ndim!=1:
        raise ValueError('Expected matching per-example loss and weight vectors')
    if values.numel()==0:raise ValueError('Empty objective term')
    return (values*weights.detach()).mean()


def forget_infonce(anchor,positive,retained_candidates,temperature):
    if temperature<=0:raise ValueError('Temperature must be positive')
    if anchor.ndim!=2 or anchor.shape!=positive.shape:
        raise ValueError('Paired views must have matching [anchors, dimensions] shapes')
    if retained_candidates.ndim!=2 or retained_candidates.shape[1]!=anchor.shape[1]:
        raise ValueError('Candidate dimension mismatch')
    # Equivalent cosine normalization with well-defined second derivatives at zero.
    normal=lambda x:x/x.square().sum(-1,keepdim=True).clamp_min(1e-24).sqrt()
    a,p,c=map(normal,(anchor,positive,retained_candidates))
    matched=(a*p).sum(-1)/temperature
    logits=torch.cat([matched[:,None],a@c.T/temperature],dim=1)
    return torch.logsumexp(logits,dim=1)-matched


def combine_losses(retained_recommendation,retained_contrastive,forget_margins,
                   forget_contrastive,weights,regularization,*,native_cl_coefficient,
                   forget_strength=.1,forget_cl_strength=.1):
    """Separate L_R and L_F for a guarded optimizer; weights are detached.

    Native retained losses must be supplied in a declared compatible reduction.
    A full-batch/alternating backbone requires its own faithful decomposition;
    wrapping its summed objective as one sample is not an automatic adaptation.
    """
    retain=weighted_mean(retained_recommendation,weights['retain_rec'])
    retain=retain+native_cl_coefficient*weighted_mean(retained_contrastive,weights['retain_cl'])+regularization
    forget=weighted_mean(F.softplus(forget_margins),weights['forget_rec'])
    forget=forget-forget_cl_strength*native_cl_coefficient*weighted_mean(forget_contrastive,weights['forget_cl'])
    return dict(retain=retain,forget=forget,total=retain+forget_strength*forget)
