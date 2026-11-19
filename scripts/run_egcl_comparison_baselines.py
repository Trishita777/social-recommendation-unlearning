"""Versioned EGCL SCIF/DA2 adaptations; never overwrites existing results."""
import argparse,copy,fcntl,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from run_unlearning import ROOT,load_data,digest,atomic_json,evaluate,seed_everything
from run_egcl_backbone import parameter_state,load_parameters,atomic_torch,synchronize,save_training,resume_training
from run_egcl_unlearning_adapters import native_data
from social_unlearning.egcl import EGCL
from social_unlearning.egcl_baselines import ProbeEGCL,scif_rhs,DA2EGCL
from social_unlearning.scif_edges import conjugate_gradient


def run_scif(original,student,data,cfg,out,device):
    normalizer=max(1,len(original.interaction_index))
    rhs=scif_rhs(original,student,normalizer)
    initial=parameter_state(student);best=-float('inf');chosen=None;trials=[]
    for damping in cfg['damping_trials']:
        load_parameters(student,initial);student.eval();p=student.user_embeddings
        loss=student.objectives()[1]/normalizer
        gradient=torch.autograd.grad(loss,p,create_graph=True)[0]
        def op(v):return torch.autograd.grad((gradient*v).sum(),p,retain_graph=True)[0].detach()+damping*v
        delta,solver=conjugate_gradient(op,rhs,cfg['cg_iterations'],1e-5)
        row=dict(damping=damping,solver=solver)
        if solver['status'] in ['converged','zero_rhs']:
            scale=min(1.,1/max(float(delta.norm()),1e-12))
            with torch.no_grad():p.sub_(scale*delta)
            metrics,_=evaluate(student,data,data.val,device);row['validation']=metrics
            if metrics['recall@20']>best:
                best=metrics['recall@20'];chosen=parameter_state(student);selected=damping
        trials.append(row);atomic_json(out/'validation_trials.json',trials)
        del gradient,loss,delta
    if chosen is None:raise RuntimeError('All SCIF solves failed; no fallback to another algorithm')
    load_parameters(student,chosen)
    return dict(trials=trials,selected_damping=selected,gradient_change_norm=float(rhs.norm()),
                normalizer=normalizer,anchor_probe=cfg['anchor_probe'],candidate_denominator='full native universe')


def run_da2(original,student,raw,deleted,olddata,data,cfg,prov,out,device):
    objective=DA2EGCL(original,student,raw,deleted,olddata,
                     ROOT/'external/baseline_sources/DA2Unlearn',cfg['seed'],cfg)
    opts=[torch.optim.Adam(student.weight_parameters(),lr=cfg['learning_rate']),
          torch.optim.Adam(student.embedding_parameters(),lr=cfg['adapter_lr'])]
    state=dict(epoch=0,best=-float('inf'),bad=0,best_epoch=0,complete=False,history=[])
    if (out/'last.pt').exists():
        state=resume_training(out/'last.pt',student,opts,cfg,prov)
        objective.eta=state['eta'];objective.ref_u=state['ref_u'].to(device)
        objective.ref_i=state['ref_i'].to(device);objective.rng.bit_generator.state=state['request_rng']
    while state['epoch']<cfg['epochs'] and not state['complete']:
        student.train();opts[0].zero_grad(set_to_none=True)
        wl=student.objectives(include_cl=False)[0]
        wp=student.weight_parameters();wg=torch.autograd.grad(wl,wp)
        if not torch.isfinite(wl) or any(not torch.isfinite(g).all() for g in wg):raise FloatingPointError('DA2 weight objective')
        for p,g in zip(wp,wg):p.grad=g
        opts[0].step();opts[1].zero_grad(set_to_none=True)
        loss=objective.loss(data);ep=state['epoch']+1
        if not torch.isfinite(loss):raise FloatingPointError('DA2 objective')
        params=student.embedding_parameters();grads=torch.autograd.grad(loss,params)
        if any(not torch.isfinite(g).all() for g in grads):raise FloatingPointError('DA2 gradients')
        for p,g in zip(params,grads):p.grad=g
        opts[1].step();objective.epoch_end(ep)
        metrics,_=evaluate(student,data,data.val,device);improved=metrics['recall@20']>state['best']
        state.update(epoch=ep,bad=0 if improved else state['bad']+1)
        if improved:state.update(best=metrics['recall@20'],best_epoch=ep)
        state['complete']=ep==cfg['epochs'] or state['bad']>=cfg['patience']
        row=dict(epoch=ep,validation=metrics,components=objective.last_components)
        state['history'].append(row);atomic_json(out/'history.json',state['history'])
        state.update(eta=objective.eta,ref_u=objective.ref_u.detach().cpu(),
                     ref_i=objective.ref_i.detach().cpu(),request_rng=objective.rng.bit_generator.state)
        if improved:save_training(out/'best_val.pt',student,opts,cfg,prov,state)
        save_training(out/'last.pt',student,opts,cfg,prov,state)
        print(json.dumps(row),flush=True)
    best=torch.load(out/'best_val.pt',map_location='cpu',weights_only=False);load_parameters(student,best['model'])
    return {k:v for k,v in state.items() if k not in ['history','ref_u','ref_i','request_rng']}


def run(a):
    torch.set_num_threads(8);device=torch.device(a.device)
    if device.type=='cuda':torch.cuda.set_device(device)
    raw=load_data(a.dataset,a.seed);split=raw.meta['split_hash'][:16]
    oldpath=ROOT/'checkpoints/EGCL'/a.dataset/f'seed_{a.seed}'/split/'original/deployment.pt'
    ck=torch.load(oldpath,map_location='cpu',weights_only=False)
    deleted=np.sort(np.random.default_rng(a.seed+10000).permutation(raw.U)[:max(1,round(raw.U*.01))]);deletion=digest(deleted)
    cfg=dict(ck['config'],seed=a.seed,method=a.method,epochs=300,patience=10,adapter_lr=.0005,
             anchor_probe=512,cg_iterations=50,damping_trials=[.01,.1,1.],
             deletion_semantics='incident_edges_only_keep_user_rows',adapter_version='egcl_scif_da2_v1')
    files=[Path(__file__),ROOT/'src/social_unlearning/egcl_baselines.py',ROOT/'src/social_unlearning/egcl.py',
           ROOT/'src/social_unlearning/baselines.py',ROOT/'src/social_unlearning/scif_edges.py',
           ROOT/'src/social_unlearning/edge_deletion.py',ROOT/'scripts/run_unlearning.py',
           ROOT/'scripts/run_egcl_backbone.py',ROOT/'scripts/run_egcl_unlearning_adapters.py']
    if a.method=='da2':files.append(ROOT/'external/baseline_sources/DA2Unlearn/assess/base/difficulty.py')
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    prov=dict(ck['provenance'],source=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),
              original_source=ck['provenance']['source'],deletion=deletion,
              original_checkpoint_sha256=hashlib.sha256(oldpath.read_bytes()).hexdigest(),
              deletion_semantics=cfg['deletion_semantics'])
    version=hashlib.sha256(json.dumps(dict(config=cfg,provenance=prov),sort_keys=True).encode()).hexdigest()[:16]
    out=ROOT/'checkpoints/egcl_comparison_baselines_v1'/a.dataset/f'seed_{a.seed}'/version/a.method;out.mkdir(parents=True,exist_ok=True)
    with (out/'run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'results.json').exists():print('REUSE',out,flush=True);return
        seed_everything(a.seed);resumed=(out/'last.pt').exists();synchronize(device)
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
        start=time.perf_counter();data=native_data(raw,deleted,device);olddata=native_data(raw,None,device)
        cls=ProbeEGCL if a.method=='scif' else EGCL
        original=cls(olddata,cfg,device);student=cls(data,cfg,device)
        load_parameters(original,ck['model']);load_parameters(student,ck['model'])
        if a.method=='scif':
            original.set_probe(a.seed,cfg['anchor_probe']);student.set_probe(a.seed,cfg['anchor_probe'])
        atomic_json(out/'manifest.json',dict(config=cfg,provenance=prov,source_files=hashes,deleted_users=deleted.tolist(),counts=data.counts,
                    adaptation='SCIF: selective native EGCL HVP with fixed anchor probe; DA2: official difficulty/schedule, full native repair, original-graph witnesses',
                    limitation='fixed-ID social adaptations, no published-certificate transfer; not contrastive proposed-method results'))
        training=run_scif(original,student,data,cfg,out,device) if a.method=='scif' else run_da2(original,student,raw,deleted,olddata,data,cfg,prov,out,device)
        atomic_torch(out/'deployment.pt',dict(model=parameter_state(student),config=cfg,provenance=prov,
                    retained_original_ids=np.arange(raw.U),graph_buffers='rebuild_from_verified_data'))
        synchronize(device);elapsed=time.perf_counter()-start
        peak=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None
        # Final diagnostics are outside the resource interval.
        refpath=ROOT/'checkpoints/edge_user_v1/EGCL'/a.dataset/f'seed_{a.seed}'/split/deletion[:16]/'retrain/deployment.pt'
        refck=torch.load(refpath,map_location='cpu',weights_only=False)
        assert refck['provenance']['deletion']==deletion and refck['provenance']['deletion_semantics']==cfg['deletion_semantics']
        reference=EGCL(data,cfg,device);load_parameters(reference,refck['model']);_,ref=evaluate(reference,data,data.test,device)
        metrics,_=evaluate(student,data,data.test,device,reference=ref)
        report=dict(status='completed_native_EGCL_adaptation',backbone='EGCL',dataset=a.dataset,seed=a.seed,fraction=.01,
                    method=a.method,config=cfg,provenance=prov,metrics=metrics,training=training,
                    efficiency=dict(wall_seconds=elapsed,peak_allocated_bytes=peak,resumed=resumed,
                                    wall_scope='current invocation; first graph through deployment; excludes final diagnostics',timing_mode='shared_GPU_not_isolated'),
                    mia_status='not_run',interpretation='EGCL social adaptation; not unchanged baseline reproduction',checkpoint=str(out/'deployment.pt'))
        atomic_json(out/'results.json',report)
        atomic_json(ROOT/'results/egcl_comparison_baselines_v1'/a.dataset/f'seed_{a.seed}'/version/a.method/'results.json',report)
        print('EGCL_BASELINE_RESULT',out/'results.json',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True,choices=['epinions','yelp2018']);p.add_argument('--seed',required=True,type=int)
    p.add_argument('--method',required=True,choices=['scif','da2']);p.add_argument('--device',default='cuda:0');run(p.parse_args())
