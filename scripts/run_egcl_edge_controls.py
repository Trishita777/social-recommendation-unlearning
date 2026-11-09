"""EGCL fresh retraining / controls after incident-edge removal, fixed ID universe."""
import argparse,hashlib,json,time
import numpy as np
import torch
from run_unlearning import ROOT,load_data,digest,seed_everything,evaluate,atomic_json
from run_egcl_backbone import EGCL,train,load_parameters,parameter_state,atomic_torch
from social_unlearning.edge_deletion import edge_graph


def run(a):
    torch.set_num_threads(8);device=torch.device(a.device);torch.cuda.set_device(device)
    raw=load_data(a.dataset,a.seed);split=raw.meta['split_hash'][:16]
    original=ROOT/'checkpoints/EGCL'/a.dataset/f'seed_{a.seed}'/split/'original/deployment.pt'
    ck=torch.load(original,map_location='cpu',weights_only=False)
    cfg=dict(ck['config'],method=a.method,deletion_semantics='incident_edges_only_keep_user_rows')
    if a.method=='retain_finetune':cfg.update(epochs=20,min_epochs=20)
    deleted=np.sort(np.random.default_rng(a.seed+10000).permutation(raw.U)[:max(1,round(raw.U*.01))]);deletion=digest(deleted)
    base=ROOT/'checkpoints/edge_user_v1/EGCL'/a.dataset/f'seed_{a.seed}'/split/deletion[:16]
    out=base/a.method;out.mkdir(parents=True,exist_ok=True)
    from pathlib import Path
    files=[Path(__file__),ROOT/'src/social_unlearning/egcl.py',ROOT/'src/social_unlearning/edge_deletion.py',ROOT/'scripts/run_egcl_backbone.py']
    source=hashlib.sha256(b''.join(p.read_bytes() for p in files)).hexdigest()
    prov=dict(ck['provenance'],source=source,original_source=ck['provenance']['source'],deletion=deletion,deletion_semantics=cfg['deletion_semantics'])
    if (out/'results.json').exists():
        result=json.loads((out/'results.json').read_text())
        if result['provenance']!=prov:raise ValueError('EGCL edge result provenance mismatch')
        print('REUSE_COMPLETED',out,flush=True);return
    seed_everything(a.seed);torch.cuda.reset_peak_memory_stats(device);start=time.perf_counter()
    data=edge_graph(raw,deleted,device);del data.adj_ui,data.adj_uu,data.norm_adj
    model=EGCL(data,cfg,device)
    if a.method!='retrain':load_parameters(model,ck['model'])
    atomic_json(base/'deletion.json',dict(deleted_users=deleted.tolist(),counts=data.counts,provenance=prov))
    if a.method=='graph_only':
        timing=dict(epoch=0,optimizer_seconds=0.,complete=True)
        atomic_torch(out/'deployment.pt',dict(model=parameter_state(model),config=cfg,provenance=prov,retained_original_ids=np.arange(raw.U)))
    else:timing=train(model,data,cfg,prov,out,device)
    torch.cuda.synchronize(device);elapsed=time.perf_counter()-start;peak=torch.cuda.max_memory_allocated(device)
    reference=None
    if a.method!='retrain':
        r=torch.load(base/'retrain/deployment.pt',map_location='cpu',weights_only=False)
        reference_model=EGCL(data,cfg,device);load_parameters(reference_model,r['model'])
        _,reference=evaluate(reference_model,data,data.test,device)
    metrics,_=evaluate(model,data,data.test,device,reference=reference)
    atomic_json(out/'results.json',dict(status='completed',method=a.method,backbone='EGCL',dataset=a.dataset,seed=a.seed,
        fraction=.01,config=cfg,provenance=prov,metrics=metrics,timing=timing,counts=data.counts,
        efficiency=dict(wall_seconds=elapsed,peak_allocated_bytes=peak,timing_mode='shared_gpu_not_isolated'),mia_status='pending_benchmark_MIO'))
    print('EGCL_EDGE_RESULT',out/'results.json',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--method',choices=['retrain','retain_finetune','graph_only'],required=True);p.add_argument('--device',default='cuda:0')
    run(p.parse_args())
