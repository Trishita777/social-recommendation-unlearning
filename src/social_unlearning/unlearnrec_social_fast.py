"""Numerically checked CSR execution for the unchanged UnlearnRec adapter objective."""
import torch
from social_unlearning.unlearnrec_social import SocialBackbone as BaseBackbone,encoder_graphs,load_official_encoder

def csr_graphs(model):
 for name in ['adj_ui','adj_uu','sparse_norm_adj']:
  value=getattr(model,name)
  if value.layout==torch.sparse_coo:setattr(model,name,value.to_sparse_csr())
 return model

class SocialBackbone(BaseBackbone):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs);csr_graphs(self.original)
 def set_request(self,data):
  super().set_request(data);csr_graphs(self.body)
