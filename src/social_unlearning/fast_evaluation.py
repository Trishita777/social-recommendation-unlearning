import torch
import numpy as np
from collections import defaultdict

@torch.no_grad()
def evaluate_fast(model,data,pairs,device,reference=None,groups=None):
    model.eval(); U,I=model(); truth=defaultdict(set)
    for u,i in pairs: truth[int(u)].add(int(i))
    totals=defaultdict(float); per={}; ranks={}; n=0
    all_users=list(truth)
    for start in range(0,len(truth),128):
        users=all_users[start:start+128]; scores=U[users]@I.T
        mask_rows=[]; mask_items=[]
        for row,u in enumerate(users):
            items=list(data.positives[u])
            mask_rows.extend([row]*len(items)); mask_items.extend(items)
        if mask_items:
            scores[torch.tensor(mask_rows,device=scores.device),torch.tensor(mask_items,device=scores.device)]=-torch.inf
        ids=scores.topk(min(20,data.item_num),dim=1).indices.cpu().numpy()
        for u,top in zip(users,ids):
            n+=1; hit=np.array([i in truth[u] for i in top]); row={}
            for k in (10,20):
                h=hit[:k]; denom=min(len(truth[u]),k)
                row[f'precision@{k}']=float(h.sum()/k); row[f'recall@{k}']=float(h.sum()/len(truth[u]))
                row[f'map@{k}']=float(((np.cumsum(h)/np.arange(1,len(h)+1))*h).sum()/denom)
                row[f'ndcg@{k}']=float((h/np.log2(np.arange(2,len(h)+2))).sum()/(1/np.log2(np.arange(2,denom+2))).sum())
            for key,value in row.items(): totals[key]+=value
            per[u]=row; ranks[u]=top
    result={key:value/max(n,1) for key,value in totals.items()}; result['evaluated_users']=n
    if groups is not None:
        result['groups']={name:dict(users=len(us),**{key:float(np.mean([per[u][key] for u in us])) for key in totals}) for name, us in {name:[u for u in per if groups[u]==name] for name in set(groups.values())}.items() if us}
    if reference is not None:
        ru,ri,rr=reference; errors=[]; agreements=[]; grouped=defaultdict(lambda: [[],[]])
        # fixed item probe; same catalog and users for all compared models
        probe=torch.arange(0,data.item_num,max(1,data.item_num//1024),device=device)
        for start in range(0,len(truth),128):
            users=list(truth)[start:start+128]; a=(U[users]@I[probe].T); b=ru[users]@ri[probe].T
            rmse=((a-b).square().mean(1).sqrt()).cpu().tolist()
            for u,e in zip(users,rmse):
                agr=len(set(ranks[u])&set(rr[u]))/len(ranks[u]); errors.append(e); agreements.append(agr)
                if groups is not None: grouped[groups[u]][0].append(e); grouped[groups[u]][1].append(agr)
        result['fidelity']=dict(score_rmse=float(np.mean(errors)),top20_overlap=float(np.mean(agreements)),score_probe_items=len(probe),groups={g:dict(users=len(x[0]),score_rmse=float(np.mean(x[0])),top20_overlap=float(np.mean(x[1]))) for g,x in grouped.items()})
    return result,(U.detach(),I.detach(),ranks)
