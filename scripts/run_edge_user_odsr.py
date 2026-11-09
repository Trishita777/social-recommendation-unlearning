"""ODSR comparison for incident-edge removal with unchanged user/item universes."""
import argparse
import copy
import fcntl
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from run_unlearning import (ROOT,load_data,graph,compact,ODSR,seed_everything,digest,
    atomic_json,evaluate,negatives,save_checkpoint,train)
from social_unlearning.edge_deletion import edge_graph,edge_sensitivity,EdgeGuardedForgetRetain,edge_da2


def scif(original,student,data,raw,cfg,out,device,seed):
    from social_unlearning.scif_edges import gradient_change,native_probe,conjugate_gradient
    rhs,probe,meta=gradient_change(original,student,data,raw,seed,512)
    initial=copy.deepcopy(student.state_dict());best=-1;best_state=None;trials=[]
    for damping in [.01,.1,1.]:
        student.load_state_dict(initial);student.eval();p=student.embedding_dict['user_emb']
        loss=native_probe(student,*probe);g=torch.autograd.grad(loss,p,create_graph=True)[0]
        def op(v):return torch.autograd.grad((g*v).sum(),p,retain_graph=True)[0].detach()+damping*v
        delta,solver=conjugate_gradient(op,rhs,50,1e-5)
        row=dict(damping=damping,solver=solver)
        if solver['status'] in ['converged','zero_rhs']:
            norm=float(delta.norm());scale=min(1.,1/max(norm,1e-12))
            with torch.no_grad():p.sub_(scale*delta)
            metrics,_=evaluate(student,data,data.val,device);row['validation']=metrics
            if metrics['recall@20']>best:best=metrics['recall@20'];best_state=copy.deepcopy(student.state_dict());selected=damping
        trials.append(row);atomic_json(out/'validation_trials.json',trials)
        del g,loss,delta
    if best_state is None:raise RuntimeError('All SCIF solver trials failed; no fallback result')
    student.load_state_dict(best_state)
    return student,dict(probe=meta,trials=trials,selected_damping=selected)


def optimize(original,student,orig,data,raw,cfg,out,prov,device,a):
    if a.method in ['retrain','retain_finetune']:
        epochs=cfg['original_epochs'] if a.method=='retrain' else cfg['unlearn_epochs']
        timing=train(student,data,cfg,out,prov,epochs,'retain_finetune',device)
        return student,timing
    if a.method=='graph_only':return student,dict(epochs=0,optimizer_seconds=0.)
    if a.method=='scif':return scif(original,student,data,raw,cfg,out,device,a.seed)
    if a.method=='gnndelete':
        from social_unlearning.gnndelete_edges import train_deletion
        return train_deletion(original,raw,data,cfg,out,device,a.seed,prov)
    if a.method=='ours':
        weights=edge_sensitivity(original,student,data,a.seed,.25)
        objective=EdgeGuardedForgetRetain(original,student,data,raw,weights,a.seed,forget_strength=.1,forget_cl_strength=.1)
        objective.forget_rec=1+.25*(objective.forget_rec-1)
        objective.forget_anchor=1+.25*(objective.forget_anchor-1)
        cfg['lr']=.0001;epochs=5
    else:
        cfg.update(eta0=1.,gamma=1.,delta=1,lambda_stability=1.,difficulty_tau=1.,lr=.0005)
        objective=edge_da2(original,student,data,raw,orig,ROOT/'external/baseline_sources/DA2Unlearn',a.seed,cfg)
        epochs=300
    optimizer=torch.optim.Adam(student.parameters(),lr=cfg['lr']);params=list(student.parameters())
    rng=np.random.default_rng(a.seed);best=-1;bad=0;history=[];steps=0;optimization=0.
    for ep in range(1,epochs+1):
        student.train();order=rng.permutation(len(data.train));start=time.perf_counter()
        for offset in range(0,len(order),cfg['batch_size']):
            ids=order[offset:offset+cfg['batch_size']];batch=data.train[ids]
            u,p=torch.as_tensor(batch,device=device).T;n=torch.as_tensor(negatives(data,batch[:,0],rng),device=device)
            optimizer.zero_grad(set_to_none=True)
            if a.method=='ours':
                objective.backward_step(student,u,p,n,torch.as_tensor(ids,device=device),params,True)
                torch.nn.utils.clip_grad_norm_(params,5,error_if_nonfinite=True)
            else:
                loss=objective(student,u,p,n,ids)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite DA2 loss')
                loss.backward()
            optimizer.step();steps+=1
        if a.method=='da2':objective.epoch_end(student,ep)
        torch.cuda.synchronize(device);optimization+=time.perf_counter()-start
        metrics,_=evaluate(student,data,data.val,device);improved=metrics['recall@20']>best
        if improved:best=metrics['recall@20'];best_state=copy.deepcopy(student.state_dict());best_epoch=ep;bad=0
        else:bad+=1
        row=dict(epoch=ep,validation=metrics,components=objective.last_components)
        history.append(row);atomic_json(out/'history.json',history)
        save_checkpoint(out/'last.pt',student,optimizer,config=cfg,provenance=prov,epoch=ep,step=steps,metrics=metrics,complete=ep==epochs or bad>=10)
        if improved:save_checkpoint(out/'best_val.pt',student,optimizer,config=cfg,provenance=prov,epoch=ep,step=steps,metrics=metrics)
        print(json.dumps(row),flush=True)
        if bad>=10:break
    student.load_state_dict(best_state)
    return student,dict(epochs=ep,best_epoch=best_epoch,steps=steps,optimizer_seconds=optimization)


def run(a):
    torch.set_num_threads(8);device=torch.device(a.device);torch.cuda.set_device(device)
    raw=load_data(a.dataset,a.seed);split=raw.meta['split_hash'][:16]
    original_path=ROOT/'checkpoints/ODSR'/a.dataset/f'seed_{a.seed}'/split/'original/deployment.pt'
    checkpoint=torch.load(original_path,map_location=device,weights_only=False)
    cfg=dict(checkpoint['config'],deletion_semantics='incident_edges_only_keep_user_rows',edge_method=a.method)
    if a.method=='ours':cfg.update(lr=.0001,epochs=5,weight_mix=.25,forget_strength=.1,forget_cl_strength=.1,project=True,gradient_clip=5)
    if a.method=='da2':cfg.update(lr=.0005,epochs=300,eta0=1.,gamma=1.,delta=1,lambda_stability=1.,difficulty_tau=1.)
    deleted=np.sort(np.random.default_rng(a.seed+10000).permutation(raw.U)[:max(1,round(raw.U*a.fraction))]);deletion=digest(deleted)
    base=ROOT/'checkpoints/edge_user_v1/ODSR'/a.dataset/f'seed_{a.seed}'/split/deletion[:16]
    out=base/a.method;out.mkdir(parents=True,exist_ok=True)
    files=[Path(__file__),ROOT/'src/social_unlearning/edge_deletion.py',ROOT/'src/social_unlearning/native_guarded_memory.py',ROOT/'src/social_unlearning/native_guarded.py',ROOT/'src/social_unlearning/baselines.py',ROOT/'src/social_unlearning/scif_edges.py',ROOT/'src/social_unlearning/gnndelete_edges.py',ROOT/'scripts/run_unlearning.py',ROOT/'external/ODSR/ODSR.py']
    source=hashlib.sha256(b''.join(p.read_bytes() for p in files)).hexdigest()
    prov=dict(checkpoint['provenance'],source=source,original_source=checkpoint['provenance']['source'],deletion=deletion,deletion_semantics=cfg['deletion_semantics'])
    with (out/'run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'results.json').exists():
            report=json.loads((out/'results.json').read_text())
            if report['provenance']!=prov:raise ValueError('Edge-user result provenance mismatch')
            print('REUSE_COMPLETED',out,flush=True);return
        orig=graph(raw,np.arange(raw.U),device);original=ODSR(orig,cfg,device).to(device);original.load_state_dict(checkpoint['model'])
        torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device);start=time.perf_counter()
        data=edge_graph(raw,deleted,device);seed_everything(a.seed)
        student=ODSR(data,cfg,device).to(device) if a.method=='retrain' else compact(original,data,cfg,device)
        atomic_json(base/'deletion.json',dict(deleted_users=deleted.tolist(),fraction=a.fraction,counts=data.counts,provenance=prov))
        student,timing=optimize(original,student,orig,data,raw,cfg,out,prov,device,a)
        atomic_payload=dict(model=student.state_dict(),retained_original_ids=np.arange(raw.U),deleted_user_ids=deleted,
            config=cfg,provenance=prov,training=timing,backbone='ODSR')
        torch.save(atomic_payload,out/'deployment.tmp');(out/'deployment.tmp').replace(out/'deployment.pt')
        torch.cuda.synchronize(device);wall=time.perf_counter()-start;peak=torch.cuda.max_memory_allocated(device)
        reference=None
        if a.method!='retrain':
            reference_path=base/'retrain/deployment.pt'
            reference_ck=torch.load(reference_path,map_location=device,weights_only=False)
            if reference_ck['provenance']['deletion_semantics']!=cfg['deletion_semantics']:raise ValueError('Wrong reference semantics')
            reference_model=ODSR(data,cfg,device).to(device);reference_model.load_state_dict(reference_ck['model'])
            _,reference=evaluate(reference_model,data,data.test,device)
        metrics,_=evaluate(student,data,data.test,device,reference=reference)
        report=dict(status='complete_edge_user_comparison',backbone='ODSR',dataset=a.dataset,seed=a.seed,fraction=a.fraction,
            method=a.method,config=cfg,provenance=prov,counts=data.counts,metrics=metrics,training=timing,
            efficiency=dict(wall_seconds=wall,peak_allocated_bytes=peak,timing_mode='shared_gpu_requires_isolated_timing'),
            mia_status='pending_benchmark_MIO',method_acceptance='new_semantics_requires_validation; old compact fidelity gate does not transfer')
        atomic_json(out/'results.json',report);print('EDGE_RESULT',out/'results.json',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--fraction',type=float,default=.01);p.add_argument('--device',default='cuda:0')
    p.add_argument('--method',choices=['retrain','retain_finetune','graph_only','ours','da2','scif','gnndelete'],required=True)
    run(p.parse_args())
