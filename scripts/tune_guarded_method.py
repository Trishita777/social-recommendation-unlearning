"""Validation-only tuning of first-order forgetting-gradient projection.
No test pairs are evaluated or loaded into selection. Research scratch models
are used for validation fidelity, never as optimization targets. Tuning cost
is separate from deployment latency. Trial artifacts are immutable per config.
"""
import argparse,copy,hashlib,itertools,json,time
import numpy as np
import torch
from run_unlearning import ROOT,load_data,graph,compact,sensitivity,ODSR,seed_everything,digest,atomic_json,evaluate,negatives,save_checkpoint
from social_unlearning.native_guarded import GuardedNativeForgetRetain


def guarded_gradients(retain,forget,params,strength,project=True):
    gr=torch.autograd.grad(retain,params,retain_graph=True,allow_unused=True)
    gf=torch.autograd.grad(forget,params,allow_unused=True)
    gr=[torch.zeros_like(p) if g is None else g for p,g in zip(params,gr)];gf=[torch.zeros_like(p) if g is None else g for p,g in zip(params,gf)]
    dot=sum((a*b).sum() for a,b in zip(gr,gf));nr=sum(a.square().sum() for a in gr)
    conflict=float(dot)<0
    if project and conflict:gf=[b-dot/(nr+1e-12)*a for a,b in zip(gr,gf)]
    nf=sum(b.square().sum() for b in gf);scale=min(1.,float((nr/(nf+1e-12)).sqrt())) if project else 1.
    for p,a,b in zip(params,gr,gf):p.grad=a+strength*scale*b
    return dict(conflict=conflict,forget_gradient_scale=scale,retain_gradient_norm=float(nr.sqrt()))


def run(a):
    torch.set_num_threads(8);device=torch.device(a.device);raw=load_data(a.dataset,42);base=ROOT/'checkpoints/ODSR'/a.dataset/'seed_42'/raw.meta['split_hash'][:16]
    ck=torch.load(base/'original/deployment.pt',map_location=device,weights_only=False);cfg=ck['config'];orig=graph(raw,np.arange(raw.U),device);original=ODSR(orig,cfg,device).to(device);original.load_state_dict(ck['model'])
    deleted=np.sort(np.random.default_rng(10042).permutation(raw.U)[:max(1,round(raw.U*.01))]);keep=np.setdiff1d(np.arange(raw.U),deleted);deletion=digest(deleted);data=graph(raw,keep,device)
    refck=torch.load(base/deletion[:16]/'random/retrain/deployment.pt',map_location=device,weights_only=False);refmodel=ODSR(data,cfg,device).to(device);refmodel.load_state_dict(refck['model']);reference_metrics,reference=evaluate(refmodel,data,data.val,device)
    initial=compact(original,data,cfg,device);initial_metrics,_=evaluate(initial,data,data.val,device,reference=reference);weights=sensitivity(original,initial,data,np.random.default_rng(42),device);del initial,refmodel
    codehash=hashlib.sha256((ROOT/'src/social_unlearning/native_guarded.py').read_bytes()+open(__file__,'rb').read()).hexdigest();directory=ROOT/'checkpoints/tuning_native_guarded'/a.dataset/'seed_42'/raw.meta['split_hash'][:16]/deletion[:16]
    trials=[dict(lr=lr,weight_mix=mix,project=True,forget_strength=.1) for lr,mix in itertools.product([.0001,.0003],[.25,1.])]+[dict(lr=.0003,weight_mix=.25,project=False,forget_strength=.1)]
    atomic_json(directory/'protocol.json',dict(status='running',selection='validation only; utility floor 98% of scratch Recall@20 and NDCG@20; fidelity must improve initial compact model score RMSE and preserve top20 overlap; eligible trial with lowest validation score RMSE',trials=trials,reference_validation=reference_metrics,initial_validation=initial_metrics,source_hash=codehash,test_evaluation='disabled',execution='tuning may share GPU; measured tuning times are not deletion efficiency',fairness='Equivalent validation trial budgets for tuned baselines remain required before competitive claims.'))
    results=[];t0=time.perf_counter()
    for number,trial in enumerate(trials):
        path=directory/f'trial_{number:02}';trialcfg=dict(cfg,**trial,method_version='native_guarded_v4',epochs=a.epochs,gradient_clip=5);prov=dict(ck['provenance'],source=codehash,deletion=deletion,trial=str(number),original_source=ck['provenance']['source'])
        if (path/'validation_result.json').exists():
            existing=json.loads((path/'validation_result.json').read_text())
            if existing['config']!=trialcfg or existing['provenance']!=prov:raise ValueError('Tuning cache provenance mismatch')
            results.append(existing);continue
        seed_everything(42);student=compact(original,data,cfg,device);student.eval()
        w={**weights,'rec':1+trial['weight_mix']*(weights['rec']-1),'anchor':1+trial['weight_mix']*(weights['anchor']-1)}
        objective=GuardedNativeForgetRetain(original,student,data,raw,w,42,forget_strength=trial['forget_strength'],forget_cl_strength=.1)
        objective.forget_rec=1+trial['weight_mix']*(objective.forget_rec-1);objective.forget_anchor=1+trial['weight_mix']*(objective.forget_anchor-1)
        opt=torch.optim.Adam(student.parameters(),lr=trial['lr']);rng=np.random.default_rng(42);history=[];best=-1.;steps=0;best_ep=0;params=list(student.parameters())
        for ep in range(1,a.epochs+1):
            student.train();order=rng.permutation(len(data.train));conflicts=0;num_batches=0
            for offset in range(0,len(order),cfg['batch_size']):
                ids=order[offset:offset+cfg['batch_size']];batch=data.train[ids];u,p=torch.as_tensor(batch,device=device).T;n=torch.as_tensor(negatives(data,batch[:,0],rng),device=device)
                opt.zero_grad(set_to_none=True);loss=objective(student,u,p,n,torch.as_tensor(ids,device=device))
                if not torch.isfinite(loss):raise ValueError('Nonfinite guarded objective')
                stats=guarded_gradients(objective.components['retain'],objective.components['forget'],params,trial['forget_strength'],trial['project']);torch.nn.utils.clip_grad_norm_(params,5,error_if_nonfinite=True);opt.step();steps+=1;num_batches+=1;conflicts+=stats['conflict']
            metrics,_=evaluate(student,data,data.val,device,reference=reference)
            # Epoch chosen using validation utility; subsequent candidate selection
            # additionally imposes the declared fidelity/utility constraints.
            score=metrics['recall@20'];improved=score>best
            if improved:best=score;best_state=copy.deepcopy(student.state_dict());best_metrics=metrics;best_ep=ep
            row=dict(epoch=ep,validation=metrics,conflict_fraction=conflicts/max(1,num_batches),last_components=objective.last_components);history.append(row);atomic_json(path/'history.json',history)
            save_checkpoint(path/'last.pt',student,opt,config=trialcfg,provenance=prov,epoch=ep,step=steps,metrics=metrics,complete=ep==a.epochs)
            if improved:save_checkpoint(path/'best_val.pt',student,opt,config=trialcfg,provenance=prov,epoch=ep,step=steps,metrics=metrics)
            print(json.dumps(dict(trial=number,**row)),flush=True)
        student.load_state_dict(best_state);torch.save(dict(model=student.state_dict(),retained_original_ids=keep,config=trialcfg,provenance=prov,best_epoch=best_ep),path/'deployment.pt')
        utility_ok=best_metrics['recall@20']>=.98*reference_metrics['recall@20'] and best_metrics['ndcg@20']>=.98*reference_metrics['ndcg@20']
        fidelity_ok=best_metrics['fidelity']['score_rmse']<=initial_metrics['fidelity']['score_rmse'] and best_metrics['fidelity']['top20_overlap']>=initial_metrics['fidelity']['top20_overlap']
        result=dict(trial=number,config=trialcfg,provenance=prov,validation=best_metrics,best_epoch=best_ep,utility_constraint_passed=utility_ok,fidelity_constraint_passed=fidelity_ok,eligible=utility_ok and fidelity_ok)
        atomic_json(path/'validation_result.json',result);results.append(result);del student,objective,opt
    eligible=[r for r in results if r['eligible']];selected=min(eligible,key=lambda r:r['validation']['fidelity']['score_rmse']) if eligible else None
    report=dict(status='selected_on_validation' if selected else 'no_trial_satisfied_declared_constraints',selected=selected,trials=results,elapsed_tuning_seconds=time.perf_counter()-t0,reference_validation=reference_metrics,initial_validation=initial_metrics,test_metrics='not_computed',next_step='freeze selected configuration before three-seed target tests; forgetting MIA remains required')
    atomic_json(directory/'selection.json',report);print('VALIDATION_SELECTION '+str(directory/'selection.json'),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',default='epinions');p.add_argument('--device',default='cuda:0');p.add_argument('--epochs',type=int,default=5);run(p.parse_args())
