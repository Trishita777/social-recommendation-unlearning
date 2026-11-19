"""Provenance-checked DiffNet++ no-attribute training and user unlearning.

Native released optimizer/loss/batches. Same splits and deletion IDs as the
ODSR study. Original-paper sampled metrics are reported separately from the
common full-catalog evaluation. Test data never selects epochs or damping.
"""
import argparse
import gc
import hashlib
import json
import os
import socket
import time
from pathlib import Path
import numpy as np
import torch
from run_unlearning import ROOT, load_data, digest, atomic_json, seed_everything, evaluate
from social_unlearning.diffnet_plus import DiffNetPlus, make_graph, compact, parameter_state, training_examples, user_batch
from social_unlearning.diffnet_influence import DiffNetObjective, solve
from social_unlearning.paper_metrics import candidate_bank, evaluate_sampled


def sync(device):
    if device.type=='cuda': torch.cuda.synchronize(device)


def atomic_save(path, payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');torch.save(payload,tmp);os.replace(tmp,path)


def source_hashes():
    files=[Path(__file__),ROOT/'src/social_unlearning/diffnet_plus.py',ROOT/'src/social_unlearning/diffnet_influence.py',
           ROOT/'src/social_unlearning/paper_metrics.py',ROOT/'src/social_unlearning/influence_spillover.py',ROOT/'scripts/run_unlearning.py',
           ROOT/'scripts/run_diffnet_plus_mia.py',ROOT/'src/social_unlearning/history_mia.py',ROOT/'src/social_unlearning/interaction_mia.py']
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def deployment(path,model,data,cfg,provenance,**extra):
    atomic_save(path,dict(model=parameter_state(model),retained_original_ids=data.keep,config=cfg,provenance=provenance,**extra))


def train(model,data,cfg,out,provenance,validation_bank,device,epochs):
    out.mkdir(parents=True,exist_ok=True)
    if (out/'deployment.pt').exists():
        ck=torch.load(out/'deployment.pt',map_location='cpu',weights_only=False)
        if ck['provenance']!=provenance or ck['config']!=cfg: raise ValueError('Cache provenance mismatch')
        model.load_state_dict(ck['model']);return json.loads((out/'timing.json').read_text())
    rng=np.random.default_rng(cfg['seed']);optimizer=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'])
    history=[];first=1;best=-float('inf');bad=0;steps=0;best_epoch=0;elapsed=0.;optimizer_seconds=0.
    if (out/'last.pt').exists():
        ck=torch.load(out/'last.pt',map_location=device,weights_only=False)
        if ck['provenance']!=provenance or ck['config']!=cfg: raise ValueError('Resume provenance mismatch')
        model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer']);rng.bit_generator.state=ck['rng']
        first=ck['epoch']+1;history=ck['history'];best=ck['best'];bad=ck['bad'];steps=ck['steps'];best_epoch=ck['best_epoch']
        elapsed=ck['elapsed'];optimizer_seconds=ck['optimizer_seconds']
        torch.set_rng_state(ck['torch_rng'].cpu())
        if device.type=='cuda':torch.cuda.set_rng_state(ck['cuda_rng'].cpu(),device)
    started=time.perf_counter()
    for epoch in range(first,epochs+1):
        if bad>=cfg['patience_evaluations']:break
        model.train();examples=training_examples(data,rng,cfg['negative_draws'])
        users=rng.permutation(sorted(examples));total=0.;samples=0
        sync(device);training_start=time.perf_counter()
        for start in range(0,len(users),cfg['users_per_batch']):
            batch=user_batch(examples,users[start:start+cfg['users_per_batch']],device)
            optimizer.zero_grad(set_to_none=True);loss=model.loss(*batch)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite DiffNet++ native loss')
            loss.backward();optimizer.step();steps+=1;total+=float(loss.detach());samples+=len(batch[0])
        sync(device);optimizer_seconds+=time.perf_counter()-training_start
        row=dict(epoch=epoch,half_squared_error_per_example=total/samples,steps=steps)
        if epoch==1 or epoch%cfg['evaluate_every']==0 or epoch==epochs:
            model.eval()
            with torch.no_grad(): metrics=evaluate_sampled(model(),data,validation_bank)
            score=metrics['metrics']['ndcg@15'];row['validation']=metrics
            if score>best:
                best=score;bad=0;best_epoch=epoch
                deployment(out/'best.pt',model,data,cfg,provenance,best_epoch=epoch)
            else:bad+=1
        row['elapsed_seconds']=elapsed+time.perf_counter()-started;history.append(row)
        atomic_save(out/'last.pt',dict(model=parameter_state(model),optimizer=optimizer.state_dict(),config=cfg,provenance=provenance,
                    rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(device) if device.type=='cuda' else None,
                    epoch=epoch,history=history,best=best,bad=bad,steps=steps,best_epoch=best_epoch,elapsed=row['elapsed_seconds'],optimizer_seconds=optimizer_seconds))
        atomic_json(out/'history.json',history)
        print(json.dumps(dict(method=out.name,**row)),flush=True)
    ck=torch.load(out/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['model'])
    timing=dict(training_seconds=elapsed+time.perf_counter()-started,optimizer_seconds=optimizer_seconds,epochs=history[-1]['epoch'],
                epoch_budget=epochs,best_epoch=best_epoch,steps=steps,convergence='validation_patience' if bad>=cfg['patience_evaluations'] else 'configured_epoch_budget')
    atomic_json(out/'timing.json',timing)
    deployment(out/'deployment.pt',model,data,cfg,provenance,best_epoch=best_epoch)
    return timing


def pilot(raw,data,cfg,device,out):
    model=DiffNetPlus(data,cfg['dimension'],device)
    examples=training_examples(data,np.random.default_rng(cfg['seed']),cfg['negative_draws'])
    batch=user_batch(examples,list(examples)[:cfg['users_per_batch']],device)
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate']);times=[]
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    for step in range(4):
        sync(device);start=time.perf_counter();optimizer.zero_grad(set_to_none=True)
        loss=model.loss(*batch);loss.backward();optimizer.step();sync(device)
        times.append(time.perf_counter()-start)
    # Profile an HVP too: training can fit while second derivatives do not.
    sync(device);start=time.perf_counter();params=tuple(model.parameters())
    grads=torch.autograd.grad(model.loss(*batch),params,create_graph=True)
    hv=torch.autograd.grad(sum((g*torch.ones_like(g)).sum() for g in grads),params)
    sync(device);hvp_seconds=time.perf_counter()-start
    report=dict(status='pilot_only_not_a_result',dataset=raw.meta['dataset'],seed=cfg['seed'],native_loss=float(loss.detach()),
                parameters=sum(p.numel() for p in params),batch_examples=len(batch[0]),step_seconds=times,hvp_seconds=hvp_seconds,
                batches_per_epoch=int(np.ceil(len(examples)/cfg['users_per_batch'])),
                projected_training_seconds_per_epoch=float(np.mean(times[1:])*np.ceil(len(examples)/cfg['users_per_batch'])),
                peak_allocated_mb=torch.cuda.max_memory_allocated(device)/2**20 if device.type=='cuda' else None,
                hvp_finite=all(bool(torch.isfinite(h).all()) for h in hv),host=socket.gethostname())
    atomic_json(out,report);print(json.dumps(report),flush=True)


def run(a):
    torch.set_num_threads(8);device=torch.device(a.device);seed_everything(a.seed)
    raw=load_data(a.dataset,a.seed);original_data=make_graph(raw,np.arange(raw.U))
    cfg=dict(seed=a.seed,dimension=64,learning_rate=.0005,users_per_batch=512,negative_draws=8,
             original_epochs=a.epochs,finetune_epochs=20,evaluate_every=5,patience_evaluations=10,
             validation_selection='sampled NDCG@15; five fixed 1000-negative draws; no test selection',
             architecture='released DiffNet++ two-layer, no side attributes',upstream_commit='cad45d58cc4667bcaa787c8601871eb18963f47a')
    hashes=source_hashes();source=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    prov=dict(split=raw.meta['split_hash'],social=raw.meta['social_hash'],source=source,
              data=hashlib.sha256(json.dumps(raw.meta['source_files'],sort_keys=True).encode()).hexdigest(),deletion='original')
    base=ROOT/'checkpoints/diffnet_plus_native_v1'/a.dataset/f'seed_{a.seed}'/source[:16]
    base.mkdir(parents=True,exist_ok=True)
    if a.pilot:
        pilot(raw,original_data,cfg,device,ROOT/'results/diffnet_plus_validation'/f'pilot_{a.dataset}_{a.seed}.json');return
    torch_report=ROOT/'results/diffnet_plus_validation/parity.torch.json'
    tf_report=ROOT/'results/diffnet_plus_validation/parity.tensorflow.json'
    if any(not p.exists() or json.loads(p.read_text())['status']!='pass' for p in (torch_report,tf_report)):
        raise RuntimeError('Both PyTorch HVP and original TensorFlow parity checks must pass before training')
    atomic_json(base/'manifest.json',dict(backbone='DiffNet++',configuration=cfg,source_files=hashes,provenance=prov,data=raw.meta,
                protocol_differences='No side attributes; current ODSR dataset splits; 3 seeds; validation-only stopping; distinct negatives exclude all observed items; all evaluator users included; this is not absolute reproduction of the original paper tables.',
                hardware=dict(host=socket.gethostname(),gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else 'cpu'),status='running'))
    validation=candidate_bank(raw,raw.val,np.arange(raw.U),a.seed+1000)
    original=DiffNetPlus(original_data,64,device)
    original_timing=train(original,original_data,cfg,base/'original',prov,validation,device,a.epochs)
    deleted=np.sort(np.random.default_rng(a.seed+10000).permutation(raw.U)[:max(1,round(raw.U*.01))])
    keep=np.setdiff1d(np.arange(raw.U),deleted)
    start=time.perf_counter();data=make_graph(raw,keep);graph_seconds=time.perf_counter()-start
    retained_prov=dict(prov,deletion=digest(deleted))
    atomic_json(base/'deletion.json',dict(deleted_users=deleted.tolist(),fraction=.01,provenance=retained_prov,graph_seconds=graph_seconds))
    val_retained=([u for u in validation[0] if data.mapping[u]>=0],validation[1],validation[2])
    test_bank=candidate_bank(raw,raw.test,keep,a.seed+2000)
    seed_everything(a.seed);retrain=DiffNetPlus(data,64,device)
    rt=train(retrain,data,cfg,base/'retrain',retained_prov,val_retained,device,a.epochs)
    rm,reference=evaluate(retrain,data,data.test,device)
    with torch.no_grad():rp=evaluate_sampled(retrain(),data,test_bank)
    records=dict(retrain=dict(metrics=rm,paper_metrics=rp,**rt,total_deletion_seconds=graph_seconds+rt['training_seconds'],checkpoint=str(base/'retrain/deployment.pt')))
    del retrain;gc.collect()
    class Restricted:
        def eval(self): return self
        def __call__(self):
            u,i=original();return u[data.keep],i
    om,_=evaluate(Restricted(),data,data.test,device,reference=reference)
    with torch.no_grad():op=evaluate_sampled(Restricted()(),data,test_bank)
    records['original']=dict(metrics=om,paper_metrics=op,**original_timing,checkpoint=str(base/'original/deployment.pt'))
    for method in ('graph_only','sorecremover','retain_finetune'):
        sync(device);start=time.perf_counter()
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
        model=compact(original,original_data,data)
        out=base/method;extra={}
        if (out/'deployment.pt').exists():
            ck=torch.load(out/'deployment.pt',map_location='cpu',weights_only=False)
            if ck['provenance']!=retained_prov or ck['config']!=cfg:raise ValueError('Method cache mismatch')
            model.load_state_dict(ck['model']);extra=json.loads((out/'timing.json').read_text())
        elif method=='sorecremover':
            rng=np.random.default_rng(a.seed+80000)
            examples=training_examples(original_data,rng,cfg['negative_draws'])
            chosen=rng.choice(sorted(examples),min(512,len(examples)),replace=False)
            retained_chosen=[int(u) for u in chosen if data.mapping[u]>=0]
            replacements=[int(u) for u in keep if u not in set(retained_chosen)]
            retained_chosen+=rng.choice(replacements,len(chosen)-len(retained_chosen),replace=False).tolist()
            retained_examples={int(data.mapping[u]):example for u,example in examples.items() if data.mapping[u]>=0}
            oldbatch=user_batch(examples,chosen,device)
            newbatch=user_batch(retained_examples,data.mapping[retained_chosen],device)
            objective=DiffNetObjective(original,model,original_data,data,oldbatch,newbatch,a.seed)
            delta,diagnostics=solve(objective,callback=lambda row:print(json.dumps(dict(method=method,solver=row)),flush=True))
            with torch.no_grad():
                for p,d in zip(objective.params,delta):p.add_(d)
            extra=dict(diagnostics=diagnostics,original_batch_users=chosen.tolist(),retained_batch_original_users=retained_chosen)
            del objective,delta,examples,retained_examples,oldbatch,newbatch;gc.collect()
        elif method=='retain_finetune':
            extra=train(model,data,cfg,out,retained_prov,val_retained,device,cfg['finetune_epochs'])
        sync(device)
        if 'total_deletion_seconds' not in extra:
            extra.update(total_deletion_seconds=graph_seconds+time.perf_counter()-start,
                         peak_allocated_mb=torch.cuda.max_memory_allocated(device)/2**20 if device.type=='cuda' else None)
        atomic_json(out/'timing.json',extra)
        if not (out/'deployment.pt').exists():deployment(out/'deployment.pt',model,data,cfg,retained_prov)
        metrics,_=evaluate(model,data,data.test,device,reference=reference)
        with torch.no_grad():paper_metrics=evaluate_sampled(model(),data,test_bank)
        records[method]=dict(metrics=metrics,paper_metrics=paper_metrics,**extra,checkpoint=str(out/'deployment.pt'))
        atomic_json(out/'results.json',records[method])
        print(json.dumps(dict(method=method,metrics=metrics,paper_metrics=paper_metrics)),flush=True)
        del model;gc.collect()
    report=dict(backbone='DiffNet++',dataset=a.dataset,seed=a.seed,fraction=.01,provenance=retained_prov,methods=records,
                status='utility_complete_mia_pending',source_files=hashes)
    atomic_json(base/'results.json',report)
    print('DIFFNET_CASE',base,flush=True)
    # Audit the exact final checkpoints with the same user-disjoint history attack.
    from run_diffnet_plus_mia import audit
    audit(base,raw,deleted,device)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['epinions','yelp2018'],required=True)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--device',default='cuda:0');p.add_argument('--epochs',type=int,default=450)
    p.add_argument('--pilot',action='store_true');run(p.parse_args())
