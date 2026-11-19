"""Native direct-and-spillover influence unlearning; compact retained deployment."""
import argparse,fcntl,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from run_unlearning import ROOT,ODSR,atomic_json,digest,evaluate,graph,load_data,negatives,seed_everything
from run_egcl_backbone import atomic_torch,load_parameters,parameter_state,synchronize
from run_egcl_unlearning_adapters import native_data
from social_unlearning.egcl_curvature import StreamedCurvatureEGCL as EGCL
from social_unlearning.influence_spillover import SurvivingObjective,solve_influence,solve_shifted_krylov

FAMILY='sorecremover_influence_spillover_v4'

def run(a):
    torch.set_num_threads(8);device=torch.device(a.device)
    if device.type=='cuda':torch.cuda.set_device(device)
    raw=load_data(a.dataset,a.seed);split=raw.meta['split_hash'][:16]
    original_path=ROOT/'checkpoints'/a.backbone/a.dataset/f'seed_{a.seed}'/split/'original/deployment.pt'
    ck=torch.load(original_path,map_location='cpu',weights_only=False)
    deleted=np.sort(np.random.default_rng(a.seed+10000).permutation(raw.U)[:max(1,round(raw.U*a.fraction))])
    keep=np.setdiff1d(np.arange(raw.U),deleted);dh=digest(deleted)
    cfg=dict(ck['config'],method='sorecremover_influence',version=FAMILY,
             damping_initial=a.damping,solver_tolerance=a.tolerance,solver_max_steps=a.max_steps,solver_max_restarts=8,
             objective_probe_batch=2048,objective_probe_batches=a.probe_batches,
             deletion_semantics='compact_user_deletion_rebuild_native_graphs',
             objective='native J0 minus native JR on surviving coordinates; deleted rows fixed only in J0',
             stochastic_approximation='ODSR fixed coupled minibatches and replayed native dropout; EGCL exact native full-batch losses',
             curvature='original-objective Hessian on surviving coordinates; native-block diagonal for alternating EGCL',
             parameter_blocks=['all'] if a.backbone=='ODSR' else ['embedding','weight'],
             selection='single full influence correction; damping escalates only for solver failure; no utility or attack selection',
             deployment_graph='compact retained graphs only; deleted rows excluded')
    paths=[Path(__file__),ROOT/'src/social_unlearning/influence_spillover.py',ROOT/'scripts/run_unlearning.py',ROOT/'src/social_unlearning/egcl.py',ROOT/'src/social_unlearning/egcl_curvature.py',ROOT/'scripts/run_egcl_backbone.py',ROOT/'external/ODSR/ODSR.py']
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    prov=dict(ck['provenance'],source=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),
              original_source=ck['provenance']['source'],deletion=dh,
              original_checkpoint_sha256=hashlib.sha256(original_path.read_bytes()).hexdigest(),deletion_semantics=cfg['deletion_semantics'])
    version=hashlib.sha256(json.dumps(dict(config=cfg,provenance=prov),sort_keys=True).encode()).hexdigest()[:16]
    out=ROOT/'checkpoints'/FAMILY/a.backbone/a.dataset/f'seed_{a.seed}'/dh[:16]/version;out.mkdir(parents=True,exist_ok=True)
    with (out/'run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'results.json').exists():print('REUSE',out,flush=True);return
        seed_everything(a.seed);synchronize(device)
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
        start=time.perf_counter();olddata=graph(raw,np.arange(raw.U),device);data=graph(raw,keep,device)
        if a.backbone=='ODSR':
            original=ODSR(olddata,cfg,device).to(device);student=ODSR(data,cfg,device).to(device)
            original.load_state_dict(ck['model'])
        else:
            for d in [olddata,data]:del d.adj_ui,d.adj_uu,d.norm_adj
            original=EGCL(olddata,cfg,device);student=EGCL(data,cfg,device);load_parameters(original,ck['model'])
        state=parameter_state(original);user_key='embedding_dict.user_emb' if a.backbone=='ODSR' else 'user_embeddings'
        state[user_key]=state[user_key][keep];load_parameters(student,state)
        original_parameters=parameter_state(original)
        batches=[]
        if a.backbone=='ODSR':
            rng=np.random.default_rng(a.seed+6301)
            for _ in range(a.probe_batches):
                # Coupled, uniformly sampled batches. Replacing each requested draw with
                # an independent retained draw gives a uniform retained minibatch.
                old=raw.train[rng.integers(len(raw.train),size=cfg['objective_probe_batch'])].copy()
                new=old.copy();mask=np.isin(new[:,0],deleted)
                replacements=data.train[rng.integers(len(data.train),size=int(mask.sum()))].copy()
                replacements[:,0]=keep[replacements[:,0]];new[mask]=replacements
                oldneg=negatives(olddata,old[:,0],rng);newneg=oldneg.copy()
                if mask.any():newneg[mask]=negatives(olddata,new[mask,0],rng)
                new[:,0]=data.mapping[new[:,0]]
                ob=tuple(torch.as_tensor(x,device=device) for x in [old[:,0],old[:,1],oldneg])
                nb=tuple(torch.as_tensor(x,device=device) for x in [new[:,0],new[:,1],newneg]);batches.append((ob,nb))
        else:batches=[(None,None)]
        assert data.user_num==raw.U-len(deleted) and not np.isin(data.keep,deleted).any()
        atomic_json(out/'manifest.json',dict(config=cfg,provenance=prov,source_files=hashes,deleted_users=deleted.tolist(),retained_original_ids=keep.tolist()))
        if a.backbone=='ODSR':
            np.savez_compressed(out/'objective_probe.npz',**{f'{side}_{k}_{j}':t.cpu().numpy() for j,pair in enumerate(batches) for side,triplet in zip(['original','retained'],pair) for k,t in zip(['user','positive','negative'],triplet)})
        diagnostics={};updates=[]
        blocks=['embedding'] if a.backbone=='ODSR' else ['embedding','weight']
        for block in blocks:
            objectives=[SurvivingObjective(original,student,keep,a.backbone,ob,nb,seed=a.seed+919+j,block=block) for j,(ob,nb) in enumerate(batches)]
            if len(objectives)==1:objective=objectives[0]
            else:
                # Average independently replayed native minibatch objectives, including
                # each minibatch's own normalization and native regularization.
                class MeanObjective:
                    params=objectives[0].params
                    def loss(self,old):return sum(o.loss(old) for o in objectives)/len(objectives)
                    gradient=SurvivingObjective.gradient
                    discrepancy=SurvivingObjective.discrepancy
                    hvp=SurvivingObjective.hvp
                objective=MeanObjective()
            def progress(row):
                print(json.dumps(dict(block=block,**row)),flush=True)
                atomic_json(out/'progress.json',dict(block=block,latest=row))
            solver=solve_influence if a.backbone=='ODSR' else solve_shifted_krylov
            delta,report=solver(objective,damping=a.damping,tolerance=a.tolerance,max_steps=a.max_steps,callback=progress)
            diagnostics[block]=report;updates.append((objective.params,delta))
            atomic_json(out/'solver.json',diagnostics)
        # Both EGCL block corrections were computed at the same original parameters.
        with torch.no_grad():
            for params,delta in updates:
                for p,v in zip(params,delta):p.add_(v)
        assert all(torch.equal(p.detach().cpu(),original_parameters[n]) for n,p in original.named_parameters())
        assert dict(student.named_parameters())[user_key].shape[0]==len(keep)
        atomic_torch(out/'deployment.pt',dict(model=parameter_state(student),config=cfg,provenance=prov,retained_original_ids=keep,deleted_user_ids=deleted,
                                             graph_buffers='rebuild_compact_retained_graph_from_verified_data'))
        validation,_=evaluate(student,data,data.val,device)
        synchronize(device);elapsed=time.perf_counter()-start
        peak=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None
        reference_path=ROOT/'checkpoints/edge_user_v1'/a.backbone/a.dataset/f'seed_{a.seed}'/split/dh[:16]/'retrain/deployment.pt'
        refck=torch.load(reference_path,map_location='cpu',weights_only=False)
        for key in ['split','social','deletion','data']:assert prov[key]==refck['provenance'][key],key
        rd=native_data(raw,deleted,device) if a.backbone=='EGCL' else __import__('social_unlearning.edge_deletion',fromlist=['edge_graph']).edge_graph(raw,deleted,device)
        reference=ODSR(rd,refck['config'],device).to(device) if a.backbone=='ODSR' else EGCL(rd,refck['config'],device)
        if a.backbone=='ODSR':reference.load_state_dict(refck['model'])
        else:load_parameters(reference,refck['model'])
        class Restricted:
            def eval(self):reference.eval();return self
            def __call__(self):
                u,i=reference();return u[torch.as_tensor(keep,device=device)],i
        _,ref=evaluate(Restricted(),data,data.test,device)
        metrics,_=evaluate(student,data,data.test,device,reference=ref)
        report=dict(status='completed',backbone=a.backbone,dataset=a.dataset,seed=a.seed,fraction=a.fraction,method=cfg['method'],family=FAMILY,
                    config=cfg,provenance=prov,metrics=metrics,counts=data.counts,solver=diagnostics,validation=validation,
                    reference_graph='retained fixed-ID retraining restricted to surviving users; unchanged historical checkpoint',
                    efficiency=dict(wall_seconds=elapsed,peak_allocated_bytes=peak,timing_mode='shared_GPU_not_isolated',wall_scope='graph reconstruction, fixed objective preparation, discrepancy gradients, all damping attempts/HVP solves, deployment and validation; excludes final test/fidelity and MIA'),
                    mia_status='pending_history_probe_audit',checkpoint=str(out/'deployment.pt'))
        atomic_json(out/'results.json',report);print('INFLUENCE_RESULT',out/'results.json',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--backbone',choices=['ODSR','EGCL'],required=True);p.add_argument('--dataset',choices=['epinions','yelp2018'],required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--fraction',type=float,default=.01);p.add_argument('--device',default='cuda:0')
    p.add_argument('--damping',type=float,default=.01);p.add_argument('--tolerance',type=float,default=1e-3);p.add_argument('--max-steps',type=int,default=50);p.add_argument('--probe-batches',type=int,default=1)
    a=p.parse_args()
    if not 0<a.fraction<1 or a.damping<=0 or a.probe_batches<1:p.error('Invalid fraction/damping/probe count')
    run(a)
