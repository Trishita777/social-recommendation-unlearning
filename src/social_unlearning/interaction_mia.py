"""User-disjoint, popularity-matched interaction membership audit utilities."""
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score


def sampling_plan(raw, deleted, seed, train_cap=5000, validation_cap=2000, negatives=64):
    rng = np.random.default_rng(seed + 73001)
    members, heldout, occupied = defaultdict(list), defaultdict(list), defaultdict(set)
    for u, i in raw.train: members[int(u)].append(int(i))
    for u, i in raw.test: heldout[int(u)].append(int(i))
    for pairs in [raw.train, raw.val, raw.test]:
        for u, i in pairs: occupied[int(u)].add(int(i))
    popularity = np.bincount(raw.train[:, 1], minlength=raw.I)
    bins = np.floor(np.log2(1 + popularity)).astype(int)
    deleted = set(map(int, deleted)); paired = {}
    for u in sorted(members):
        left, right = defaultdict(list), defaultdict(list)
        for i in members[u]: left[int(bins[i])].append(i)
        for i in heldout[u]: right[int(bins[i])].append(i)
        common = sorted(set(left) & set(right))
        if not common: continue
        b = int(rng.choice(common))
        pair = [int(rng.choice(left[b])), int(rng.choice(right[b]))]
        assert pair[0] != pair[1]
        paired[u] = pair
    retained = rng.permutation(sorted(set(paired) - deleted))
    ntrain = min(train_cap, max(1, int(len(retained)*.7)))
    roles = dict(train=np.sort(retained[:ntrain]),
                 validation=np.sort(retained[ntrain:ntrain+validation_cap]),
                 target=np.array(sorted(set(paired) & deleted), dtype=np.int64))
    if min(map(len, roles.values())) < 10: raise ValueError('Too few eligible users for a membership audit')
    users = np.concatenate(list(roles.values())); items = np.array([paired[int(u)] for u in users])
    neg = []
    for u in users:
        chosen = set()
        if raw.I-len(occupied[int(u)]) < negatives: raise ValueError('Insufficient negative catalog')
        while len(chosen) < negatives:
            i = int(rng.integers(raw.I))
            if i not in occupied[int(u)]: chosen.add(i)
        neg.append(sorted(chosen))
    return dict(users=users, items=items, negatives=np.array(neg, dtype=np.int64),
                roles={k:v.tolist() for k,v in roles.items()}, requested_users=len(deleted),
                eligible_requested_users=len(roles['target']),
                popularity_bins=bins[items], seed=seed)


def features(user_vectors, item_vectors, plan):
    """ID-free embedding/score features; same sampled negatives for every model."""
    out=[]
    for start in range(0,len(plan['users']),256):
        ids=plan['users'][start:start+256]; u=user_vectors[ids].astype(np.float64)
        i=item_vectors[plan['items'][start:start+256]].astype(np.float64)
        neg=item_vectors[plan['negatives'][start:start+256]].astype(np.float64)
        score=np.einsum('bd,bkd->bk',u,i); background=np.einsum('bd,bnd->bn',u,neg)
        un=np.linalg.norm(u,axis=1)[:,None]; inn=np.linalg.norm(i,axis=2)
        mean=background.mean(1)[:,None]; std=background.std(1)[:,None]
        rank=(score[:,:,None]>background[:,None,:]).mean(2)
        arr=np.stack([score,score/np.maximum(un*inn,1e-12),np.broadcast_to(un,score.shape),inn,
                      score-mean,(score-mean)/np.maximum(std,1e-8),rank,
                      np.broadcast_to(std,score.shape)],axis=-1)
        out.append(arr)
    x=np.concatenate(out)
    if not np.isfinite(x).all(): raise FloatingPointError('Invalid membership features')
    return x


def metrics(paired_probabilities, threshold):
    p=np.asarray(paired_probabilities); y=np.tile([1,0],len(p)); flat=p.ravel()
    if p.ndim!=2 or p.shape[1]!=2 or not np.isfinite(p).all(): raise ValueError('Expected paired probabilities')
    tpr=float((p[:,0]>=threshold).mean());fpr=float((p[:,1]>=threshold).mean())
    return dict(auc=float(roc_auc_score(y,flat)),tpr=tpr,fpr=fpr,balanced_accuracy=(tpr+1-fpr)/2,
                threshold=float(threshold),member_interactions=len(p),nonmember_interactions=len(p),users=len(p))


def paired_bootstrap(p, seed, repeats=500):
    rng=np.random.default_rng(seed+381);values=[]
    for _ in range(repeats):
        sample=p[rng.integers(len(p),size=len(p))]
        values.append(roc_auc_score(np.tile([1,0],len(sample)),sample.ravel()))
    return dict(auc_95ci=np.quantile(values,[.025,.975]).tolist(),bootstrap='paired whole-user',samples=repeats)
