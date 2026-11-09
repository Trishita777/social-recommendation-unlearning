"""Explicit isolated-row reset control; not a shared-influence guarantee."""
import math
import numpy as np
import torch


@torch.no_grad()
def reset_isolated_odsr_rows(model, data, deleted, seed):
    deleted=np.asarray(deleted,dtype=np.int64)
    if len(np.unique(deleted))!=len(deleted) or (deleted<0).any() or (deleted>=data.user_num).any():
        raise ValueError('Invalid requested rows')
    if data.ui[deleted].nnz or np.isin(data.social,deleted).any():
        raise ValueError('Reset control requires isolated requested users')
    adjacency=data.adj_uu.coalesce()
    ids=torch.as_tensor(deleted,device=adjacency.device)
    if torch.isin(adjacency.indices(),ids).any():
        raise ValueError('Requested rows still have social adjacency entries')
    table=model.embedding_dict['user_emb']
    if table.shape[0]!=data.user_num:
        raise ValueError('Full user table required')
    bound=math.sqrt(6/(table.shape[0]+table.shape[1]))
    generator=torch.Generator(device='cpu').manual_seed(seed)
    rows=torch.empty((len(deleted),table.shape[1]),dtype=table.dtype,device='cpu')
    rows.uniform_(-bound,bound,generator=generator)
    table[torch.as_tensor(deleted,device=table.device)]=rows.to(table.device)
    return dict(seed=seed,distribution='uniform_original_full_table_xavier',bound=bound,
        rows=len(deleted),limitation='Resets isolated user memory only; shared learned influence remains')
