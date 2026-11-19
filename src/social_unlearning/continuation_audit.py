"""Shadow-development membership audit, disjoint from benchmark request users."""
import copy
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from .history_mia import history_plan
from .interaction_mia import metrics, paired_bootstrap


def benchmark_requests(users, seed):
    return np.sort(np.random.default_rng(seed+10000).permutation(users)[:max(1, round(users*.01))])


def shadow_requests(users):
    reserved = np.unique(np.concatenate([benchmark_requests(users, s) for s in [42, 43, 44]]))
    pool = np.setdiff1d(np.arange(users), reserved)
    shadow = np.sort(np.random.default_rng(916247).permutation(pool)[:max(1, round(users*.01))])
    assert not np.intersect1d(shadow, reserved).size
    return shadow, reserved


def development_plan(raw, shadow, reserved, seed):
    # Never use raw.test as development membership labels or candidate targets.
    dev = copy.copy(raw)
    dev.test = raw.val.copy()
    dev.val = np.empty((0, 2), dtype=np.int64)
    plan = history_plan(dev, np.union1d(shadow, reserved), seed+901)
    wanted = set(plan['roles']['train'] + plan['roles']['validation']) | set(map(int, shadow))
    idx = [k for k, u in enumerate(plan['users']) if int(u) in wanted]
    oldptr = plan['history_indptr']; olditems = plan['history_items']
    chunks = [olditems[oldptr[k]:oldptr[k+1]] for k in idx]
    for field in ['users', 'items', 'negatives', 'popularity_bins']:
        plan[field] = plan[field][idx]
    plan['history_items'] = np.concatenate(chunks)
    plan['history_indptr'] = np.r_[0, np.cumsum([len(c) for c in chunks])]
    plan['roles']['target'] = [u for u in plan['roles']['target'] if u in set(map(int, shadow))]
    plan['eligible_requested_users'] = len(plan['roles']['target'])
    assert len(plan['roles']['target']) >= 20, 'Insufficient shadow membership coverage'
    assert not set(plan['users']) & set(map(int, reserved))
    return plan


def fit_attack(x, plan, seed):
    nt = len(plan['roles']['train']); nv = len(plan['roles']['validation'])
    train, val = x[:nt].reshape(-1,8), x[nt:nt+nv].reshape(-1,8)
    yt, yv = np.tile([1,0],nt), np.tile([1,0],nv)
    candidates = [(f'logistic_C={c}', make_pipeline(StandardScaler(), LogisticRegression(C=c,max_iter=1000,random_state=seed))) for c in [.01,.1,1.,10.]]
    candidates.append(('hist_gradient_boosting', HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=7,l2_regularization=1.,early_stopping=False,random_state=seed)))
    best = None; trials = []
    for name, model in candidates:
        model.fit(train,yt); prob = model.predict_proba(val)[:,1]; auc = float(roc_auc_score(yv,prob))
        trials.append(dict(name=name,validation_auc=auc))
        if best is None or auc > best[0]: best = (auc,name,model,prob)
    auc,name,model,prob = best
    threshold = float(np.nextafter(np.quantile(prob[yv==0],.95,method='higher'),np.inf))
    return model,dict(name=name,validation_auc=auc,threshold=threshold,trials=trials)


def audit_attack(model, fit, x, plan, seed, intervals=False):
    target = len(plan['roles']['train']) + len(plan['roles']['validation'])
    prob = model.predict_proba(x[target:].reshape(-1,8))[:,1].reshape(-1,2)
    result = dict(metrics=metrics(prob,fit['threshold']),fit=fit)
    if intervals: result['uncertainty'] = paired_bootstrap(prob,seed)
    return result,prob


def detectability(*aucs):
    """Two-sided point estimate; below-chance reversal is not a privacy gain."""
    return max(max(float(a),1-float(a)) for a in aucs)
