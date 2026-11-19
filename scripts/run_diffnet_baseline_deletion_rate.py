"""Fresh source-audited DiffNet++ adaptations; final metrics and frozen MIA."""
import argparse,copy,gc,hashlib,json,socket,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.nn import functional as F
import joblib
from run_unlearning import ROOT,load_data,digest,atomic_json,seed_everything,evaluate
from run_diffnet_plus import atomic_save,sync
from run_native_ablations import one,sha
from social_unlearning.diffnet_plus import DiffNetPlus,make_graph,compact,parameter_state,training_examples,user_batch
from social_unlearning.diffnet_baselines import GNNDeleteDiffNet,DA2DiffNet,deletion_targets
from social_unlearning.rrl_social import RRL
from social_unlearning.influence_spillover import cg,norm
from social_unlearning.paper_metrics import candidate_bank,evaluate_sampled
from social_unlearning.history_mia import history_plan,history_features
from social_unlearning.interaction_mia import metrics,paired_bootstrap


class Restricted:
    def __init__(self,model,keep):self.model,self.keep=model,keep
    def eval(self):self.model.eval();return self
    def __call__(self):
        u,i=self.model();return u[self.keep],i


def run(a):
    torch.set_num_threads(6);device=torch.device(a.device)
    if device.type=='cuda':torch.cuda.set_device(device)
    raw=load_data(a.dataset,a.seed)
    base=one(p.parent for p in (ROOT/'checkpoints/diffnet_plus_deletion_rate_v1'/f'fraction_{a.fraction:g}'/a.dataset/f'seed_{a.seed}').glob('*/results.json') if json.loads(p.read_text())['status']=='complete_utility_and_mia')
    parent=json.loads((base/'results.json').read_text());ck=torch.load(base/'original/deployment.pt',map_location='cpu',weights_only=False)
    deleted=np.array(json.loads((base/'deletion.json').read_text())['deleted_users']);keep=np.setdiff1d(np.arange(raw.U),deleted)
    paths=[Path(__file__),ROOT/'src/social_unlearning/diffnet_baselines.py',ROOT/'src/social_unlearning/diffnet_plus.py',
           ROOT/'src/social_unlearning/rrl_social.py',ROOT/'src/social_unlearning/baselines.py',ROOT/'src/social_unlearning/gnndelete_edges.py',
           ROOT/'src/social_unlearning/influence_spillover.py',ROOT/'src/social_unlearning/paper_metrics.py',ROOT/'src/social_unlearning/history_mia.py',
           ROOT/'src/social_unlearning/interaction_mia.py',ROOT/'external/baseline_sources/DA2Unlearn/assess/base/difficulty.py',
           ROOT/'external/baseline_sources/GNNDelete/framework/models/deletion.py']
    cfg=dict(method=a.method,dimension=64,seed=a.seed,selection='retained validation sampled NDCG@15; no target MIA selection',
             scif_damping_trials=[.01,.1,1.],scif_max_steps=50,scif_tolerance=1e-5,scif_user_batch=512,scif_max_update_norm=1.,
             da2_epochs=300,da2_lr=.0005,da2_validation_every=5,da2_patience_checks=10,
             rrl_epochs=20,rrl_lr=.001,rrl_fisher_samples=64,rrl_forget_batch=2048,
             gnndelete_steps=1500,gnndelete_lr=.001,gnndelete_validation_every=25,gnndelete_patience_checks=10)
    identity=dict(fraction=a.fraction,backbone='DiffNet++',dataset=a.dataset,seed=a.seed,method=a.method,config=cfg,
                  original_checkpoint_sha256=sha(base/'original/deployment.pt'),reference_checkpoint_sha256=sha(base/'retrain/deployment.pt'),
                  split=raw.meta['split_hash'],deletion=digest(deleted),source_files={str(p.relative_to(ROOT)):sha(p) for p in paths},
                  graph='compact_retained' if a.method=='gnndelete' else 'original_graph_persistent_ids')
    version=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16]
    out=ROOT/'checkpoints/diffnet_baselines_deletion_rate_v1'/f'fraction_{a.fraction:g}'/a.dataset/f'seed_{a.seed}'/a.method/version;out.mkdir(parents=True,exist_ok=True)
    if (out/'results.json').exists():print('REUSE_BASELINE',out,flush=True);return
    atomic_json(out/'manifest.json',dict(identity,status='running',host=socket.gethostname(),interpretation='Declared social adaptation, not unchanged published baseline reproduction'))
    seed_everything(a.seed);olddata=make_graph(raw,np.arange(raw.U));data=make_graph(raw,keep)
    original=DiffNetPlus(olddata,64,device);original.load_state_dict(ck['model'])
    student=DiffNetPlus(olddata,64,device);student.load_state_dict(ck['model'])
    validation=candidate_bank(raw,raw.val,keep,a.seed+1000)
    rng=np.random.default_rng(a.seed)
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    sync(device);start=time.perf_counter();history=[];best=-float('inf');best_state=None;bad=0;best_step=0
    def validate(step,details):
        nonlocal best,best_state,bad,best_step
        model=student if a.method=='gnndelete' else Restricted(student,keep)
        model.eval()
        with torch.no_grad():m=evaluate_sampled(model(),data,validation)
        score=m['metrics']['ndcg@15'];improved=score>best
        if improved:best=score;best_state=parameter_state(student);best_step=step;bad=0
        else:bad+=1
        history.append(dict(step=step,validation=m,**details));atomic_json(out/'history.json',history)
        print(json.dumps(dict(method=a.method,step=step,validation_ndcg15=score,**details)),flush=True)
        if improved:atomic_save(out/'best.pt',dict(model=best_state,config=cfg,provenance=identity))
        return bad
    if a.method=='scif':
        for p in student.parameters():p.requires_grad_(False)
        p=student.user_embeddings;p.requires_grad_(True)
        examples=training_examples(olddata,rng,8);chosen=rng.choice(sorted(examples),min(512,len(examples)),replace=False)
        retained=[int(u) for u in chosen if u not in set(deleted)]
        retained+=rng.choice([u for u in keep if u not in set(retained)],len(chosen)-len(retained),replace=False).tolist()
        ob=user_batch(examples,chosen,device);rb=user_batch(examples,retained,device)
        go=torch.autograd.grad(student.loss(*ob)/len(ob[0]),p)[0].detach()
        gr=torch.autograd.grad(student.loss(*rb)/len(rb[0]),p)[0].detach();rhs=(go-gr,)
        initial=p.detach().clone();trials=[]
        for k,damping in enumerate(cfg['scif_damping_trials']):
            with torch.no_grad():p.copy_(initial)
            g=torch.autograd.grad(student.loss(*rb)/len(rb[0]),p,create_graph=True)[0]
            def op(v):return (torch.autograd.grad((g*v[0]).sum(),p,retain_graph=True)[0].detach()+damping*v[0],)
            delta,report=cg(op,rhs,tolerance=1e-5,max_steps=50);trials.append(dict(damping=damping,solver=report))
            if report['converged']:
                scale=min(1.,1/max(norm(delta),1e-12))
                with torch.no_grad():p.add_(delta[0],alpha=scale)
                validate(k+1,dict(damping=damping,update_scale=scale))
            del g,delta
        if best_state is None:raise RuntimeError('No converged SCIF candidate; no fallback method substituted')
        training=dict(trials=trials,scope='user table only; native per-example mean MSE; original graph',best_trial=best_step)
    elif a.method=='rrl':
        objective=RRL(original,student,raw,deleted,['user_embeddings','item_embeddings'],a.seed,64)
        optimizer=torch.optim.SGD(objective.params,lr=.001)
        for ep in range(1,21):
            order=rng.permutation(len(objective.pairs))
            for offset in range(0,len(order),2048):
                optimizer.zero_grad(set_to_none=True);loss=objective.loss(objective.pairs[order[offset:offset+2048]])
                if not torch.isfinite(loss):raise FloatingPointError('RRL loss')
                loss.backward();optimizer.step()
            validate(ep,dict(components=objective.last_components))
        training=dict(epochs=20,best_epoch=best_step,fisher_samples=64,scope='released reverse-log-sigmoid objective and sampled low-rank Fisher; original graph')
    elif a.method=='da2':
        objective=DA2DiffNet(original,student,raw,deleted,ROOT/'external/baseline_sources/DA2Unlearn',a.seed)
        train_data=copy.copy(olddata);train_data.positives=[p if u not in set(deleted) else set() for u,p in enumerate(olddata.positives)]
        optimizer=torch.optim.Adam(student.parameters(),lr=.0005)
        for ep in range(1,301):
            examples=training_examples(train_data,rng,8);users=rng.permutation(sorted(examples))
            for offset in range(0,len(users),512):
                batch=user_batch(examples,users[offset:offset+512],device)
                optimizer.zero_grad(set_to_none=True);loss=objective.loss(batch)
                if not torch.isfinite(loss):raise FloatingPointError('DA2 loss')
                loss.backward();optimizer.step()
            objective.epoch_end(ep)
            if ep==1 or ep%5==0:
                if validate(ep,dict(components=objective.last_components))>=10:break
        training=dict(epochs=ep,epoch_budget=300,best_epoch=best_step,scope='official difficulty and adaptive schedule; original graph; native per-example mean MSE repair')
    else:
        masks,retained_masks,forgotten,negative,locality=deletion_targets(raw,deleted,data,a.seed)
        body=compact(original,olddata,data);student=GNNDeleteDiffNet(body,retained_masks,ROOT/'external/baseline_sources/GNNDelete')
        witness=GNNDeleteDiffNet(original,masks,ROOT/'external/baseline_sources/GNNDelete',student.operators)
        with torch.no_grad():teacher=student(return_layers=True,apply=False)
        fp=torch.as_tensor(forgotten,device=device).T;npair=torch.as_tensor(negative,device=device).T
        locality=[torch.as_tensor(m,device=device) for m in locality]
        optimizer=torch.optim.Adam(student.operators.parameters(),lr=.001)
        for step in range(1,1501):
            optimizer.zero_grad(set_to_none=True);ws=witness(return_layers=True);rs=student(return_layers=True)
            dec=sum(F.mse_loss(torch.cat([z[fp[0]],z[fp[1]]]),torch.cat([t[npair[0]],t[npair[1]]])) for z,t in zip(ws,teacher))
            ni=sum(F.mse_loss(z[m],t[m]) if m.any() else z.sum()*0 for z,t,m in zip(rs,teacher,locality));loss=.5*(dec+ni)
            if not torch.isfinite(loss):raise FloatingPointError('GNNDelete loss')
            loss.backward();optimizer.step()
            if step%25==0:
                if validate(step,dict(randomness=float(dec.detach()),locality=float(ni.detach())))>=10:break
        training=dict(steps=step,step_budget=1500,best_step=best_step,scope='official deletion operators after each diffusion layer; original graph witnesses, compact retained targets/locality/inference; raw skip embeddings frozen')
    with torch.no_grad():
        for name,p in student.named_parameters():p.copy_(best_state[name].to(device))
    path=out/'deployment.pt';atomic_save(path,dict(model=parameter_state(student),config=cfg,provenance=identity,
        retained_original_ids=keep if a.method=='gnndelete' else np.arange(raw.U)))
    sync(device);wall=time.perf_counter()-start;peak=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None
    reference=DiffNetPlus(data,64,device);reference.load_state_dict(torch.load(base/'retrain/deployment.pt',map_location='cpu',weights_only=False)['model']);_,ref=evaluate(reference,data,data.test,device)
    model=student if a.method=='gnndelete' else Restricted(student,keep)
    met,emb=evaluate(model,data,data.test,device,reference=ref)
    test=candidate_bank(raw,raw.test,keep,a.seed+2000);sampled=evaluate_sampled(emb[:2],data,test)
    plan=history_plan(raw,deleted,a.seed);target=slice(len(plan['roles']['train'])+len(plan['roles']['validation']),None)
    x,_=history_features(emb[1].cpu().numpy(),plan);fit=json.loads((base/'mia/attacker.json').read_text())
    assert fit['original_checkpoint_sha256']==identity['original_checkpoint_sha256']
    attacker=joblib.load(base/'mia/attacker.joblib');prob=attacker.predict_proba(x[target].reshape(-1,8))[:,1].reshape(-1,2)
    audit=dict(metrics=metrics(prob,fit['threshold']),uncertainty=paired_bootstrap(prob,a.seed),checkpoint_sha256=sha(path),
        original_checkpoint_sha256=identity['original_checkpoint_sha256'],attacker_sha256=sha(base/'mia/attacker.joblib'),eligible_requested_users=len(prob))
    np.savez_compressed(out/'audit.npz',target_features=x[target],target_probabilities=prob)
    result=dict(status='complete',**identity,metrics=met,paper_metrics=sampled,mia=audit,training=training,checkpoint=str(path),
                efficiency=dict(wall_seconds=wall,peak_allocated_bytes=peak,scope='update, validation and deployment; excludes graph setup, final test/fidelity/MIA; shared hardware'),host=socket.gethostname())
    atomic_json(out/'results.json',result);print('BASELINE_COMPLETE',out,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--fraction',type=float,default=.10);p.add_argument('--method',choices=['scif','rrl','da2','gnndelete'],required=True);p.add_argument('--device',default='cuda:0');run(p.parse_args())
