"""Matched ablations for the CURRENT ODSR/DiffNet++ influence method only."""
import argparse,copy,gc,hashlib,json,socket,time
from pathlib import Path
import numpy as np
import torch
import joblib
from run_unlearning import ROOT,ODSR,load_data,graph,compact as compact_odsr,evaluate,atomic_json,digest,seed_everything
from run_diffnet_plus import atomic_save,sync
from social_unlearning.influence_spillover import SurvivingObjective,NativeObjective
from social_unlearning.influence_ablations import component_gradients,shared_damping_solve
from social_unlearning.diffnet_plus import DiffNetPlus,make_graph,compact as compact_diffnet,training_examples,user_batch,parameter_state
from social_unlearning.diffnet_influence import DiffNetObjective,Loss
from social_unlearning.paper_metrics import candidate_bank,evaluate_sampled
from social_unlearning.history_mia import history_plan,history_features
from social_unlearning.interaction_mia import metrics,paired_bootstrap


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def one(paths):
    paths=list(paths)
    if len(paths)!=1:raise ValueError(f'Expected one source, got {paths}')
    return paths[0]


def frozen_attacker(raw,deleted,seed,original_path,backbone,base):
    directory=base/'mia' if backbone=='DiffNet++' else ROOT/'checkpoints/interaction_mia_history_influence_v4/ODSR'/raw.meta['dataset']/f'seed_{seed}'
    candidates=[directory/'attacker.json'] if backbone=='DiffNet++' else list(directory.glob('*/attacker.json'))
    path=one(p for p in candidates if p.exists() and json.loads(p.read_text())['original_checkpoint_sha256']==sha(original_path))
    fit=json.loads(path.read_text());return joblib.load(path.with_name('attacker.joblib')),fit,path


def run(a):
    torch.set_num_threads(a.threads);device=torch.device(a.device)
    if device.type=='cuda':torch.cuda.set_device(device)
    raw=load_data(a.dataset,a.seed);split=raw.meta['split_hash'][:16]
    deleted=np.sort(np.random.default_rng(a.seed+10000).permutation(raw.U)[:max(1,round(raw.U*.01))]);keep=np.setdiff1d(np.arange(raw.U),deleted)
    if a.backbone=='ODSR':
        source=one((ROOT/'checkpoints/sorecremover_influence_spillover_v4/ODSR'/a.dataset/f'seed_{a.seed}').glob('*/ */results.json'.replace(' ','')))
        main=json.loads(source.read_text());base=source.parent
        original_path=ROOT/'checkpoints/ODSR'/a.dataset/f'seed_{a.seed}'/split/'original/deployment.pt'
        refpath=ROOT/'checkpoints/edge_user_v1/ODSR'/a.dataset/f'seed_{a.seed}'/split/digest(deleted)[:16]/'retrain/deployment.pt'
        mu=main['solver']['embedding']['solver']['damping']
    else:
        source=one(p for p in (ROOT/'checkpoints/diffnet_plus_native_v1'/a.dataset/f'seed_{a.seed}').glob('*/results.json')
                   if json.loads(p.read_text())['status']=='complete_utility_and_mia')
        main=json.loads(source.read_text());base=source.parent;original_path=base/'original/deployment.pt';refpath=base/'retrain/deployment.pt'
        mu=main['methods']['sorecremover']['diagnostics']['solver']['damping']
    attacker,fit,attacker_path=frozen_attacker(raw,deleted,a.seed,original_path,a.backbone,base)
    ck=torch.load(original_path,map_location='cpu',weights_only=False);cfg=ck['config']
    files=[Path(__file__),ROOT/'src/social_unlearning/influence_ablations.py',ROOT/'src/social_unlearning/influence_spillover.py',
           ROOT/'src/social_unlearning/diffnet_influence.py',ROOT/'src/social_unlearning/diffnet_plus.py',ROOT/'src/social_unlearning/history_mia.py',
           ROOT/'src/social_unlearning/interaction_mia.py',ROOT/'src/social_unlearning/paper_metrics.py',ROOT/'external/ODSR/ODSR.py']
    identity=dict(protocol='native_influence_matched_components_v1',backbone=a.backbone,dataset=a.dataset,seed=a.seed,
        original_checkpoint_sha256=sha(original_path),reference_checkpoint_sha256=sha(refpath),primary_result_sha256=sha(source),
        source_files={str(p.relative_to(ROOT)):sha(p) for p in files},split=raw.meta['split_hash'],deletion=digest(deleted),
        variants=['graph_only','full','examples_only','graph_change_only','gradient_norm_matched'])
    version=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16]
    out=ROOT/'checkpoints/native_influence_ablations_v1'/a.backbone/a.dataset/f'seed_{a.seed}'/version;out.mkdir(parents=True,exist_ok=True)
    if (out/'results.json').exists():print('REUSE_ABLATIONS',out,flush=True);return
    atomic_json(out/'manifest.json',dict(identity,host=socket.gethostname(),device=str(device),status='running',
        interpretation='Examples/sampling change and graph change decompose the native discrepancy through a fixed-original-graph intermediate. Shared damping is selected for all component solves. Norm-matched gradient is a direction control, not an independent cheap algorithm.'))
    seed_everything(a.seed);start=time.perf_counter()
    if a.backbone=='ODSR':
        olddata=graph(raw,np.arange(raw.U),device);data=graph(raw,keep,device)
        original=ODSR(olddata,cfg,device).to(device);original.load_state_dict(ck['model'])
        student=compact_odsr(original,data,cfg,device)
        probe=np.load(base/'objective_probe.npz')
        ob=tuple(torch.as_tensor(probe[f'original_{k}_0'],device=device) for k in ['user','positive','negative'])
        nb=tuple(torch.as_tensor(probe[f'retained_{k}_0'],device=device) for k in ['user','positive','negative'])
        objective=SurvivingObjective(original,student,keep,'ODSR',ob,nb,seed=a.seed+919)
        middle_batch=(torch.as_tensor(keep,device=device)[nb[0]],nb[1],nb[2])
        intermediate=NativeObjective(original,'ODSR',middle_batch)
    else:
        olddata=make_graph(raw,np.arange(raw.U));data=make_graph(raw,keep)
        original=DiffNetPlus(olddata,64,device);original.load_state_dict(ck['model']);student=compact_diffnet(original,olddata,data)
        rng=np.random.default_rng(a.seed+80000);examples=training_examples(olddata,rng,8)
        chosen=rng.choice(sorted(examples),min(512,len(examples)),replace=False)
        retained_chosen=[int(u) for u in chosen if data.mapping[u]>=0]
        replacements=[int(u) for u in keep if u not in set(retained_chosen)]
        retained_chosen+=rng.choice(replacements,len(chosen)-len(retained_chosen),replace=False).tolist()
        ex={int(data.mapping[u]):v for u,v in examples.items() if data.mapping[u]>=0}
        ob=user_batch(examples,chosen,device);nb=user_batch(ex,data.mapping[retained_chosen],device)
        objective=DiffNetObjective(original,student,olddata,data,ob,nb,a.seed)
        intermediate=Loss(original,(torch.as_tensor(keep,device=device)[nb[0]],nb[1],nb[2]))
    before=parameter_state(original);initial=parameter_state(student)
    rhs,decomposition=component_gradients(objective,intermediate)
    updates,solver=shared_damping_solve(objective,rhs,mu,callback=lambda x:print(json.dumps(x),flush=True))
    sync(device);setup_seconds=time.perf_counter()-start
    names=objective.names;deltas={variant:{n:v.detach().cpu() for n,v in zip(names,delta)} for variant,delta in updates.items()}
    del objective,intermediate,updates,rhs;gc.collect()
    assert all(torch.equal(v,before[n]) for n,v in parameter_state(original).items())
    atomic_json(out/'solver.json',dict(decomposition=decomposition,**solver))
    refck=torch.load(refpath,map_location='cpu',weights_only=False)
    if a.backbone=='ODSR':
        from social_unlearning.edge_deletion import edge_graph
        rd=edge_graph(raw,deleted,device);reference=ODSR(rd,refck['config'],device).to(device);reference.load_state_dict(refck['model'])
        class Restricted:
            def eval(self):reference.eval();return self
            def __call__(self):
                u,i=reference();return u[keep],i
        _,ref=evaluate(Restricted(),data,data.test,device)
        bank=None
    else:
        reference=DiffNetPlus(data,64,device);reference.load_state_dict(refck['model']);_,ref=evaluate(reference,data,data.test,device)
        bank=candidate_bank(raw,raw.test,keep,a.seed+2000)
    del reference,refck;gc.collect()
    plan=history_plan(raw,deleted,a.seed);target=slice(len(plan['roles']['train'])+len(plan['roles']['validation']),None)
    records={}
    for variant in identity['variants']:
        state={n:p+deltas[variant].get(n,0) for n,p in initial.items()};student.load_state_dict(state)
        if not all(torch.isfinite(p).all() for p in student.parameters()):raise FloatingPointError(variant)
        path=out/variant/'deployment.pt'
        atomic_save(path,dict(model=state,config=cfg,retained_original_ids=keep,provenance=identity,variant=variant))
        met,emb=evaluate(student,data,data.test,device,reference=ref)
        sampled=evaluate_sampled(emb[:2],data,bank) if bank is not None else None
        x,_=history_features(emb[1].cpu().numpy(),plan)
        prob=attacker.predict_proba(x[target].reshape(-1,8))[:,1].reshape(-1,2)
        audit=dict(metrics=metrics(prob,fit['threshold']),uncertainty=paired_bootstrap(prob,a.seed),
                   original_checkpoint_sha256=identity['original_checkpoint_sha256'],checkpoint_sha256=sha(path),
                   attacker_sha256=sha(attacker_path.with_name('attacker.joblib')),requested_users=plan['requested_users'],eligible_requested_users=plan['eligible_requested_users'])
        np.savez_compressed(path.parent/'audit.npz',target_probabilities=prob,target_features=x[target])
        row=dict(variant=variant,metrics=met,paper_metrics=sampled,mia=audit,update_norm=solver['update_norms'][variant],checkpoint=str(path))
        atomic_json(path.parent/'results.json',row);records[variant]=row
        print(json.dumps(dict(variant=variant,recall20=met['recall@20'],mia_auc=audit['metrics']['auc'])),flush=True)
    atomic_json(out/'results.json',dict(status='complete',**identity,records=records,solver=solver,decomposition=decomposition,
                shared_setup_seconds=setup_seconds,total_seconds=time.perf_counter()-start,host=socket.gethostname(),device=str(device)))
    print('ABLATIONS_COMPLETE',out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--backbone',choices=['ODSR','DiffNet++'],required=True);p.add_argument('--dataset',required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--device',default='cpu');p.add_argument('--threads',type=int,default=4);run(p.parse_args())
