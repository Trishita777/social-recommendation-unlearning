import sys
from pathlib import Path
import torch
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from social_unlearning.benchmark_mio import embedding_features, train_mio


def test_features_and_missing_row_guard():
    u=torch.tensor([[1.,2.],[3.,4.]])
    i=torch.tensor([[2.,4.],[4.,8.],[0.,1.]])
    x=embedding_features(u,i,[8,3],{3:0,8:1},{3:[2],8:[0,1]})
    torch.testing.assert_close(x,torch.tensor([[3.,4.,3.,6.],[1.,2.,0.,1.]]))
    with pytest.raises(ValueError,match='lack native embeddings'):
        embedding_features(u,i,[99],{3:0,8:1},{99:[0]})


def test_fit_is_reproducible_frozen_and_test_independent():
    torch.set_num_threads(2)
    g=torch.Generator().manual_seed(9)
    x=torch.randn(32,8,generator=g); y=torch.arange(32)%2
    vx=torch.randn(16,8,generator=g); vy=torch.arange(16)%2
    model,report=train_mio(x,y,vx,vy,seed=3,epochs=3,batch_size=8)
    other,_=train_mio(x,y,vx,vy,seed=3,epochs=3,batch_size=8)
    p=model.member_probability(vx)
    torch.testing.assert_close(p,other.member_probability(vx),rtol=0,atol=0)
    assert torch.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    assert not any(p.requires_grad for p in model.parameters())
    assert report['selection']=='shadow_validation_cross_entropy'
