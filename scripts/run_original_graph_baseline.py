"""Original-graph optimization and inference for FT, SCIF, DA2 and RRL.

Reuses the existing loss/update implementations, preserving their budgets.
Only the declared propagation/inference convention changes; results are new.
GNNDelete is excluded because its source uses mixed graph contexts.
"""
import argparse,fcntl,hashlib,json,time
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import torch
from run_unlearning import ROOT,ODSR,load_data,graph,digest,evaluate,atomic_json,seed_everything
from run_egcl_backbone import parameter_state,load_parameters,atomic_torch,synchronize,train as egcl_train
from run_edge_user_odsr import optimize as odsr_optimize
from run_egcl_comparison_baselines import run_scif,run_da2
from run_egcl_unlearning_adapters import native_data
from social_unlearning.edge_deletion import edge_graph
from social_unlearning.egcl import EGCL
from social_unlearning.egcl_baselines import ProbeEGCL
from social_unlearning.rrl_social import RRL


class RetainedContrast:
    """Filter user loss supports without changing propagation matrices."""
    retained_user_ids=None
    def info_nce(self,a,b,anchor_weights=None):
        if self.retained_user_ids is not None and len(a)==self.user_num:
            ids=self.retained_user_ids;a=a[ids];b=b[ids]
            if anchor_weights is not None:anchor_weights=anchor_weights[ids]
        return super().info_nce(a,b,anchor_weights)


class RetainedEGCL(RetainedContrast,EGCL):pass
class RetainedProbeEGCL(RetainedContrast,ProbeEGCL):
    def set_probe(self,seed,size=512):
        super().set_probe(seed,size)
        if self.retained_user_ids is not None:
            n=len(self.retained_user_ids)
            self.probes[n]=torch.as_tensor(np.sort(np.random.default_rng(seed).choice(n,min(n,size),replace=False)),device=self.device)


def run(a):
    torch.set_num_threads(8);device=torch.device(a.device);torch.cuda.set_device(device)
    raw=load_data(a.dataset,a.seed);split=raw.meta['split_hash'][:16]
    original_path=ROOT/'checkpoints'/a.backbone/a.dataset/f'seed_{a.seed}'/split/'original/deployment.pt'
    ck=torch.load(original_path,map_location='cpu',weights_only=False)
    deleted=np.sort(np.random.default_rng(a.seed+10000).permutation(raw.U)[:max(1,round(raw.U*.01))]);dh=digest(deleted)
    method=a.method+'_original_graph'
    cfg=dict(ck['config'],seed=a.seed,method=method,version='original_graph_baselines_v1',
             deletion_semantics='parameter_unlearning_original_graph_keep_user_rows',
             propagation_graph='original during optimization, validation and final predictions',
             checkpoint_selection='retained-user validation Recall@20 on original graph')
    if a.method=='retain_finetune':cfg.update(epochs=20,min_epochs=20)
    if a.method=='scif':cfg.update(anchor_probe=512,cg_iterations=50,damping_trials=[.01,.1,1.])
    if a.method=='da2':cfg.update(epochs=300,patience=10,adapter_lr=.0005)
    if a.method=='da2' and a.backbone=='ODSR':
        cfg.update(lr=.0005,eta0=1.,gamma=1.,delta=1,lambda_stability=1.,difficulty_tau=1.)
    if a.method=='rrl':cfg.update(epochs=5 if a.backbone=='ODSR' else 20,adapter_lr=.001,fisher_samples=64,batch_size=2048)
    files=[Path(__file__),ROOT/'scripts/run_edge_user_odsr.py',ROOT/'scripts/run_egcl_comparison_baselines.py',
           ROOT/'scripts/run_unlearning.py',ROOT/'scripts/run_egcl_backbone.py',ROOT/'src/social_unlearning/egcl_baselines.py',
           ROOT/'src/social_unlearning/edge_deletion.py',ROOT/'src/social_unlearning/egcl.py',ROOT/'src/social_unlearning/rrl_social.py',
           ROOT/'src/social_unlearning/scif_edges.py',ROOT/'src/social_unlearning/baselines.py',ROOT/'external/ODSR/ODSR.py']
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    prov=dict(ck['provenance'],source=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),
              original_source=ck['provenance']['source'],deletion=dh,deletion_semantics=cfg['deletion_semantics'],
              original_checkpoint_sha256=hashlib.sha256(original_path.read_bytes()).hexdigest())
    version=hashlib.sha256(json.dumps(dict(config=cfg,provenance=prov),sort_keys=True).encode()).hexdigest()[:16]
    out=ROOT/'checkpoints/original_graph_baselines_v1'/a.backbone/a.dataset/f'seed_{a.seed}'/method/version
    out.mkdir(parents=True,exist_ok=True)
    with (out/'run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'results.json').exists():print('REUSE',out,flush=True);return
        seed_everything(a.seed);synchronize(device);torch.cuda.reset_peak_memory_stats(device);start=time.perf_counter()
        if a.backbone=='ODSR':
            old=graph(raw,np.arange(raw.U),device);data=edge_graph(raw,deleted,device)
            original=ODSR(old,cfg,device).to(device);student=ODSR(old,cfg,device).to(device)
            for model in [original,student]:model.load_state_dict(ck['model'])
        else:
            old=native_data(raw,None,device);data=native_data(raw,deleted,device)
            cls=RetainedProbeEGCL if a.method=='scif' else RetainedEGCL
            original=cls(old,cfg,device);student=cls(old,cfg,device)
            for model in [original,student]:load_parameters(model,ck['model'])
            student.interaction_index=torch.as_tensor(np.column_stack(data.ui.nonzero()),device=device)
            student.social_index=torch.as_tensor(data.social,device=device)
            student.retained_user_ids=torch.as_tensor(data.retained_users,device=device)
            if a.method=='scif':
                original.set_probe(a.seed,cfg['anchor_probe']);student.set_probe(a.seed,cfg['anchor_probe'])
        atomic_json(out/'manifest.json',dict(config=cfg,provenance=prov,source_files=hashes,deleted_users=deleted.tolist(),
                    interpretation='Declared social formulation on original graphs; not an unchanged published implementation'))
        if a.method=='rrl':
            names=['embedding_dict.user_emb','embedding_dict.item_emb'] if a.backbone=='ODSR' else ['user_embeddings','item_embeddings']
            objective=RRL(original,student,raw,deleted,names,a.seed,64)
            optimizer=torch.optim.SGD(objective.params,lr=.001);rng=np.random.default_rng(a.seed);best=-float('inf');history=[]
            for epoch in range(1,cfg['epochs']+1):
                order=rng.permutation(len(objective.pairs))
                for offset in range(0,len(order),cfg['batch_size']):
                    optimizer.zero_grad(set_to_none=True);loss=objective.loss(objective.pairs[order[offset:offset+cfg['batch_size']]])
                    if not torch.isfinite(loss):raise FloatingPointError('RRL objective')
                    loss.backward()
                    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in objective.params):raise FloatingPointError('RRL gradient')
                    optimizer.step()
                metrics,_=evaluate(student,data,data.val,device)
                if metrics['recall@20']>best:
                    best=metrics['recall@20'];best_epoch=epoch;atomic_torch(out/'best_val.pt',dict(model=parameter_state(student)))
                history.append(dict(epoch=epoch,validation=metrics,components=objective.last_components));atomic_json(out/'history.json',history)
            load_parameters(student,torch.load(out/'best_val.pt',map_location='cpu',weights_only=False)['model'])
            timing=dict(epochs=cfg['epochs'],best_epoch=best_epoch)
        elif a.backbone=='ODSR':
            student,timing=odsr_optimize(original,student,old,data,raw,cfg,out,prov,device,SimpleNamespace(method=a.method,seed=a.seed))
        elif a.method=='retain_finetune':timing=egcl_train(student,data,cfg,prov,out,device)
        elif a.method=='scif':timing=run_scif(original,student,data,cfg,out,device)
        else:timing=run_da2(original,student,raw,deleted,old,data,cfg,prov,out,device)
        atomic_torch(out/'deployment.pt',dict(model=parameter_state(student),config=cfg,provenance=prov,
                     retained_original_ids=np.arange(raw.U),deleted_user_ids=deleted,graph_buffers='rebuild_original_graph_from_verified_data'))
        synchronize(device);elapsed=time.perf_counter()-start;peak=torch.cuda.max_memory_allocated(device)
        refpath=ROOT/'checkpoints/edge_user_v1'/a.backbone/a.dataset/f'seed_{a.seed}'/split/dh[:16]/'retrain/deployment.pt'
        refck=torch.load(refpath,map_location='cpu',weights_only=False)
        for key in ['split','social','deletion','data']:assert prov[key]==refck['provenance'][key],key
        reference=ODSR(data,refck['config'],device).to(device) if a.backbone=='ODSR' else EGCL(data,refck['config'],device)
        if a.backbone=='ODSR':reference.load_state_dict(refck['model'])
        else:load_parameters(reference,refck['model'])
        _,ref=evaluate(reference,data,data.test,device);metrics,_=evaluate(student,data,data.test,device,reference=ref)
        report=dict(status='completed',backbone=a.backbone,dataset=a.dataset,seed=a.seed,fraction=.01,method=method,
                    config=cfg,provenance=prov,metrics=metrics,training=timing,counts=data.counts,propagation_counts=old.counts,
                    counts_scope='retained supervision and reference graph, not propagation graph',
                    efficiency=dict(wall_seconds=elapsed,peak_allocated_bytes=peak,timing_mode='shared_GPU_not_isolated',
                                    wall_scope='graph construction, update, validation and deployment; excludes final test fidelity'),
                    mia_status='not_run',checkpoint=str(out/'deployment.pt'))
        atomic_json(out/'results.json',report);print('ORIGINAL_GRAPH_BASELINE_RESULT',out/'results.json',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--backbone',required=True,choices=['ODSR','EGCL'])
    p.add_argument('--dataset',required=True,choices=['epinions','yelp2018']);p.add_argument('--seed',required=True,type=int)
    p.add_argument('--method',required=True,choices=['retain_finetune','scif','da2','rrl']);p.add_argument('--device',default='cuda:0')
    run(p.parse_args())
