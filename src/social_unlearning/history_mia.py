"""Common interaction-MIA query for models that exclude deleted user rows.

The query user is the mean of deployed item vectors for their supplied original
training history, excluding both audited items. No deleted ID or embedding is
queried. The exact same query construction applies to originals and baselines.
"""
from collections import defaultdict
import numpy as np
from .interaction_mia import sampling_plan,features


def history_plan(raw,deleted,seed,**kwargs):
    plan=sampling_plan(raw,deleted,seed,**kwargs)
    histories=defaultdict(set)
    for u,i in raw.train:histories[int(u)].add(int(i))
    chosen=[];flat=[];indptr=[0]
    for k,(u,pair) in enumerate(zip(plan['users'],plan['items'])):
        history=sorted(histories[int(u)]-set(map(int,pair)))
        if not history:continue
        chosen.append(k);flat.extend(history);indptr.append(len(flat))
    valid=set(map(int,plan['users'][chosen]))
    for field in ['users','items','negatives','popularity_bins']:plan[field]=plan[field][chosen]
    plan['roles']={key:[u for u in users if u in valid] for key,users in plan['roles'].items()}
    if min(map(len,plan['roles'].values()))<10:raise ValueError('Too few users with query-independent histories')
    plan['history_items']=np.asarray(flat,dtype=np.int64);plan['history_indptr']=np.asarray(indptr,dtype=np.int64)
    plan['eligible_requested_users']=len(plan['roles']['target'])
    return plan


def history_features(item_vectors,plan):
    query=[]
    for a,b in zip(plan['history_indptr'][:-1],plan['history_indptr'][1:]):
        query.append(item_vectors[plan['history_items'][a:b]].astype(np.float64).mean(0))
    query=np.asarray(query)
    local=dict(plan,users=np.arange(len(plan['users'])))
    return features(query,item_vectors,local),query
