import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from types import SimpleNamespace
import numpy as np
from social_unlearning.interaction_mia import sampling_plan, features, metrics, paired_bootstrap


def test_disjoint_membership_and_popularity_matching():
    # Every item occurs equally in training; train and test edges never overlap.
    raw=SimpleNamespace(U=100,I=150,train=np.array([(u,u) for u in range(100)]),
        test=np.array([(u,(u+1)%100) for u in range(100)]),val=np.empty((0,2),dtype=int))
    p=sampling_plan(raw,np.arange(20),7,train_cap=40,validation_cap=20,negatives=10)
    a,b,c=map(set,p['roles'].values());assert not(a&b or a&c or b&c)
    assert c==set(range(20)) and not(a&set(range(20)))
    assert np.all(p['popularity_bins'][:,0]==p['popularity_bins'][:,1])
    for u,(member,nonmember),neg in zip(p['users'],p['items'],p['negatives']):
        assert member==u and nonmember==(u+1)%100
        assert not(set(neg)&{member,nonmember})
    x=features(np.ones((100,4)),np.ones((150,4)),p)
    assert x.shape==(80,2,8) and np.isfinite(x).all()


def test_zero_output_is_chance_not_evidence_of_erasure():
    p=np.full((30,2),.3)
    assert metrics(p,.5)['auc']==.5
    assert paired_bootstrap(p,3,repeats=10)['auc_95ci']==[.5,.5]
    assert metrics(np.tile([.9,.1],(30,1)),.5)['auc']==1
