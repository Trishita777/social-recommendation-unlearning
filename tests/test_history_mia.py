import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from social_unlearning.history_mia import history_plan,history_features


def test_history_probe_excludes_both_queries_and_needs_no_deleted_rows():
    raw=SimpleNamespace(U=100,I=150,train=np.array([(u,i) for u in range(100) for i in [u,(u+1)%100,(u+2)%100]]),test=np.array([(u,(u+3)%100) for u in range(100)]),val=np.empty((0,2),dtype=int))
    p=history_plan(raw,np.arange(20),42,train_cap=40,validation_cap=20,negatives=10)
    train=set(map(tuple,raw.train));test=set(map(tuple,raw.test))
    for k,(u,pair) in enumerate(zip(p['users'],p['items'])):
        history=p['history_items'][p['history_indptr'][k]:p['history_indptr'][k+1]]
        assert len(history) and not set(history)&set(pair)
        assert (u,pair[0]) in train and (u,pair[1]) in test
        assert all((u,i) in train for i in history)
    item=np.random.default_rng(7).normal(size=(150,4))
    x,q=history_features(item,p)
    assert x.shape==(80,2,8) and q.shape==(80,4)
    # Equal item vectors make both labels indistinguishable, including query norms.
    x,_=history_features(np.ones((150,4)),p)
    np.testing.assert_array_equal(x[:,0],x[:,1])
