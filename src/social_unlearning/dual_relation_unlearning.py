"""Deletion-masked contrastive targets for interaction and social relations.

This version uses neither influence functions nor reverse-ranking losses.
Targets come from the immutable learned checkpoint on the original graph,
with requested users masked out of the target candidate distribution.
The current student's original-graph views expose withdrawn candidates;
the original graph is also used for validation and deployment.
"""
import numpy as np
import torch
from torch import nn
from torch.func import functional_call
from torch.nn import functional as F


def relation_views(model, backbone):
    if backbone == 'EGCL':
        user, item, social = model.views()
        return ((user, torch.sparse.mm(model.rating_neighbors, item)),
                (social, torch.sparse.mm(model.social_neighbors, social)))
    if backbone != 'ODSR':
        raise ValueError(backbone)
    user = model.embedding_dict['user_emb']
    item = model.embedding_dict['item_emb']
    social = user
    users, items, socials = [user], [item], []
    for layer in model.layers:
        side = torch.sparse.mm(model.sparse_norm_adj, torch.cat([user, item]))
        next_item = side[model.user_num:]
        user, social, _ = layer(item, social, model.adj_ui, model.adj_uu, None, model.eps)
        item = next_item
        users.append(user); items.append(item); socials.append(social)
    weights = model.alphas.softmax(0)
    user = sum(w * x for w, x in zip(weights, users))
    item = sum(w * x for w, x in zip(weights, items))
    social = torch.stack(socials).mean(0)
    return ((user, torch.sparse.mm(model.adj_ui, item)),
            (social, torch.sparse.mm(model.adj_uu, social)))


class _Relations(nn.Module):
    def __init__(self, model, backbone):
        super().__init__(); self.model = model; self.backbone = backbone

    def forward(self):
        return relation_views(self.model, self.backbone)


def masked_target_kl(anchor, candidates, target_anchor, target_candidates,
                     anchor_ids, retained_ids, temperature=.2, soft_mix=.5):
    """KL(q_retained || p_original) with full original user denominator.

    q mixes same-user positives and a frozen original-graph similarity
    distribution. Deleted columns have exactly zero target probability.
    Its gradient still includes them in the student's full denominator.
    """
    if not 0 <= soft_mix <= 1 or temperature <= 0:
        raise ValueError('Invalid target mixture or temperature')
    logp = F.log_softmax(anchor @ candidates.T / temperature, dim=1)
    with torch.no_grad():
        q = F.softmax(target_anchor @ target_candidates.T / temperature, dim=1) * soft_mix
        positions = torch.searchsorted(retained_ids, anchor_ids)
        if not torch.equal(retained_ids[positions], anchor_ids):
            raise ValueError('A contrastive anchor is not retained')
        q[torch.arange(len(q), device=q.device), positions] += 1 - soft_mix
    # Zero-mass columns are omitted from the sum, never from logp's denominator.
    return (torch.xlogy(q, q) - q * logp[:, retained_ids]).sum(1).mean()


class DualRelationUnlearning:
    def __init__(self, original, student, data, backbone, seed, *,
                 probe_size=512, temperature=.2, soft_mix=.5, chunk=128):
        self.original = original
        self.student = student
        self.backbone = backbone
        self.temperature = temperature
        self.soft_mix = soft_mix
        self.chunk = chunk
        self.wrapper = _Relations(original, backbone)
        self.retained = torch.as_tensor(np.sort(data.retained_users), device=student.device)
        self.deleted = torch.as_tensor(data.deleted_users, device=student.device)
        self.anchors = []
        self.targets = []
        self.last_components = {}
        for p in original.parameters():
            p.requires_grad_(False)
        original.eval()
        rng = np.random.default_rng(seed + 2718)
        # Capture only frozen vectors; do not retain autograd graphs or N x N arrays.
        with torch.no_grad():
            for anchor, candidate in relation_views(original, backbone):
                eligible = self.retained[(anchor[self.retained].norm(dim=1) > 1e-8)
                                         & (candidate[self.retained].norm(dim=1) > 1e-8)]
                if not len(eligible):
                    raise ValueError('No nonzero retained anchors for a relation')
                selected = rng.choice(eligible.cpu().numpy(), min(probe_size, len(eligible)), replace=False)
                ids = torch.as_tensor(np.sort(selected), device=student.device)
                self.anchors.append(ids)
                self.targets.append((F.normalize(anchor[ids], dim=1).detach().clone(),
                                     F.normalize(candidate[self.retained], dim=1).detach().clone()))

    def losses(self):
        self.original.eval()  # deterministic witness; substituted student parameters carry gradients
        params = {'model.' + name: value for name, value in self.student.named_parameters()}
        pairs = functional_call(self.wrapper, params, ())
        losses = []
        for (a, b), ids, (ta, tb) in zip(pairs, self.anchors, self.targets):
            a = F.normalize(a, dim=1); b = F.normalize(b, dim=1)
            value = a.new_zeros(())
            for start in range(0, len(ids), self.chunk):
                ix = ids[start:start + self.chunk]
                value = value + len(ix) / len(ids) * masked_target_kl(
                    a[ix], b, ta[start:start + self.chunk], tb, ix, self.retained,
                    self.temperature, self.soft_mix)
            losses.append(value)
        self.last_components = dict(interaction_contrast=float(losses[0].detach()),
                                    social_contrast=float(losses[1].detach()))
        return tuple(losses)

    @torch.no_grad()
    def diagnostics(self):
        params = {'model.' + name: value for name, value in self.student.named_parameters()}
        pairs = functional_call(self.wrapper, params, ())
        result = {}
        for name, (a, b), ids in zip(['interaction', 'social'], pairs, self.anchors):
            a = F.normalize(a, dim=1); b = F.normalize(b, dim=1)
            masses = []
            for start in range(0, len(ids), self.chunk):
                prob = (a[ids[start:start+self.chunk]] @ b.T / self.temperature).softmax(1)
                masses.append(prob[:, self.deleted].sum(1))
            result[name + '_deleted_candidate_mass'] = float(torch.cat(masses).mean())
        return result
