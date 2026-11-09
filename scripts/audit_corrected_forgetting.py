"""Direct deleted-user score audits on corrected full-ID checkpoints, no training."""
import argparse,hashlib,json,time
from collections import defaultdict
import numpy as np
import torch
from scipy.stats import wasserstein_distance
from run_unlearning import ROOT,load_data,graph,digest,ODSR,atomic_json
from social_unlearning.edge_deletion import edge_graph
from social_unlearning.egcl import EGCL
from run_egcl_backbone import load_parameters


@torch.no_grad()
def representations(backbone,method,raw,deleted,path,device):
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    if method!='original' and checkpoint['provenance'].get('deletion_semantics')!='incident_edges_only_keep_user_rows':
        raise ValueError('Refusing compact or incompatible checkpoint')
    data=graph(raw,np.arange(raw.U),device) if method=='original' else edge_graph(raw,deleted,device)
    cfg=checkpoint['config']
    if backbone=='ODSR':
        model=ODSR(data,cfg,device).to(device)
        if method=='gnndelete':
            from social_unlearning.gnndelete_edges import GNNDeleteODSR,neighborhoods
            masks,_,_=neighborhoods(raw,deleted,model.n_layers)
            model=GNNDeleteODSR(model,masks,ROOT/'external/baseline_sources/GNNDelete')
        model.load_state_dict(checkpoint['model'])
    else:
        model=EGCL(data,cfg,device);load_parameters(model,checkpoint['model'])
    model.eval();users,items=model()
    if len(users)!=raw.U:raise ValueError('Native user rows missing')
    result=users.cpu(),items.cpu()
    return result,checkpoint['provenance']


def directory(backbone,dataset,seed,raw,deleted):
    return ROOT/'checkpoints/edge_user_v1'/backbone/dataset/f'seed_{seed}'/raw.meta['split_hash'][:16]/digest(deleted)[:16]


def paths(backbone,dataset,seed,raw,deleted):
    base=directory(backbone,dataset,seed,raw,deleted)
    result={'original':ROOT/'checkpoints'/backbone/dataset/f'seed_{seed}'/raw.meta['split_hash'][:16]/'original/deployment.pt'}
    for name in ['retrain','graph_only','retain_finetune','ours','scif','gnndelete','da2']:
        path=base/name/'deployment.pt'
        if path.exists() and (path.parent/'results.json').exists():result[name]=path
    return result


def probes(raw,deleted,seed,cap=32,negatives=128):
    by=defaultdict(list);known=defaultdict(set)
    for u,i in raw.train:by[int(u)].append(int(i));known[int(u)].add(int(i))
    for pairs in [raw.val,raw.test]:
        for u,i in pairs:known[int(u)].add(int(i))
    output=[]
    for user in deleted:
        rng=np.random.default_rng(seed*100000+int(user));positive=np.array(sorted(by[int(user)]))
        if not len(positive):continue
        positive=rng.choice(positive,min(cap,len(positive)),replace=False)
        available=np.setdiff1d(np.arange(raw.I),np.fromiter(known[int(user)],dtype=np.int64))
        if len(available)==0:raise ValueError('No valid audit negatives')
        negative=rng.choice(available,min(negatives,len(available)),replace=False)
        output.append((int(user),positive,negative))
    return output


def score_probe(embeddings,probe):
    users,items=embeddings;records=[]
    for user,positive,negative in probe:
        pos=(items[positive]@users[user]).numpy();neg=(items[negative]@users[user]).numpy()
        difference=pos[:,None]-neg[None,:]
        records.append(dict(user=user,pos=pos,neg=neg,mean_positive_score=float(pos.mean()),
            mean_pair_margin=float(difference.mean()),mean_pair_bpr_loss=float(np.logaddexp(0,-difference).mean()),
            positive_beats_negative=float(((difference>0)+.5*(difference==0)).mean())))
    return records


def compare(records,reference):
    rmse=[];w1=[];margin_gap=[]
    for a,b in zip(records,reference):
        if a['user']!=b['user']:raise ValueError('Audit probe alignment mismatch')
        x=np.concatenate([a['pos'],a['neg']]);y=np.concatenate([b['pos'],b['neg']])
        rmse.append(float(np.sqrt(np.mean((x-y)**2))));w1.append(float(wasserstein_distance(x,y)))
        margin_gap.append(abs(a['mean_pair_margin']-b['mean_pair_margin']))
    return dict(deleted_user_score_rmse=float(np.mean(rmse)),deleted_user_score_wasserstein=float(np.mean(w1)),
        deleted_user_absolute_margin_gap=float(np.mean(margin_gap)))


def run(a):
    torch.set_num_threads(4);device=torch.device(a.device);raw=load_data(a.dataset,a.seed)
    deleted=np.sort(np.random.default_rng(a.seed+10000).permutation(raw.U)[:max(1,round(raw.U*a.fraction))])
    available=paths(a.backbone,a.dataset,a.seed,raw,deleted)
    if 'retrain' not in available:raise RuntimeError('Corrected retraining must finish before audit')
    probe=probes(raw,deleted,a.seed);emb,refprov=representations(a.backbone,'retrain',raw,deleted,available['retrain'],device)
    reference=score_probe(emb,probe);del emb
    report=dict(backbone=a.backbone,dataset=a.dataset,seed=a.seed,fraction=a.fraction,
        deletion_semantics='incident_edges_only_keep_user_rows',requested_users=len(deleted),evaluated_users=len(probe),
        probe=dict(max_forget_interactions_per_user=32,negative_items_per_user=128,weighting='equal user weight',negative_exclusions='all observed train/validation/test interactions',seed=a.seed),
        reference_provenance=refprov,rows=[],interpretation='Direct native deleted-user prediction comparison to scratch; neither score suppression nor score matching proves unlearning. MIO is separate.')
    for method,path in available.items():
        start=time.perf_counter();emb,prov=representations(a.backbone,method,raw,deleted,path,device)
        records=score_probe(emb,probe);del emb
        values={k:float(np.mean([r[k] for r in records])) for k in ['mean_positive_score','mean_pair_margin','mean_pair_bpr_loss','positive_beats_negative']}
        report['rows'].append(dict(method=method,**values,**compare(records,reference),provenance=prov,audit_seconds=time.perf_counter()-start))
    report['source']=hashlib.sha256(open(__file__,'rb').read()).hexdigest()
    out=ROOT/'results/forgetting_edge_user'/a.backbone/a.dataset/f'seed_{a.seed}'/f'fraction_{a.fraction:g}'/'results.json'
    atomic_json(out,report);print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--backbone',choices=['ODSR','EGCL'],default='ODSR');p.add_argument('--dataset',required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--fraction',type=float,default=.01);p.add_argument('--device',default='cuda:0')
    run(p.parse_args())
