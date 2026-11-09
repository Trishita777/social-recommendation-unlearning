"""Method-matched shadow training for currently implemented social adapters."""
import copy,json,time
import numpy as np
import torch
from social_unlearning.checkpoint import save_checkpoint


def fit(model,data,cfg,out,provenance,epochs,device,objective=None,guard=False,epoch_end=None):
    from run_unlearning import atomic_json,evaluate,negatives,loss_terms
    out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():
        saved=torch.load(out/'deployment.pt',map_location=device,weights_only=False)
        if saved['config']!=cfg or saved['provenance']!=provenance:raise ValueError('Shadow cache provenance/config mismatch')
        model.load_state_dict(saved['model']);return
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=cfg['lr']);rng=np.random.default_rng(cfg['seed']);best=-1.;bad=0;steps=0;history=[];params=[p for p in model.parameters() if p.requires_grad]
    for ep in range(1,epochs+1):
        model.train();order=rng.permutation(len(data.train));total=0.
        for start in range(0,len(order),cfg['batch_size']):
            ids=order[start:start+cfg['batch_size']];pairs=data.train[ids];u,p=torch.as_tensor(pairs,device=device).T;n=torch.as_tensor(negatives(data,pairs[:,0],rng),device=device);optimizer.zero_grad(set_to_none=True)
            loss=loss_terms(model,u,p,n) if objective is None else objective(model,u,p,n,torch.as_tensor(ids,device=device))
            if not torch.isfinite(loss):raise ValueError('Nonfinite shadow objective')
            if guard:
                from tune_guarded_method import guarded_gradients
                guarded_gradients(objective.components['retain'],objective.components['forget'],params,cfg['forget_strength'],cfg['project']);torch.nn.utils.clip_grad_norm_(params,5,error_if_nonfinite=True)
            else:loss.backward()
            optimizer.step();steps+=1;total+=float(loss.detach())*len(ids)
        if epoch_end is not None:epoch_end(model,ep)
        metrics,_=evaluate(model,data,data.val,device);improved=metrics['recall@20']>best
        if improved:best=metrics['recall@20'];best_state=copy.deepcopy(model.state_dict());best_ep=ep;bad=0
        else:bad+=1
        row=dict(epoch=ep,loss=total/len(order),validation=metrics);history.append(row);atomic_json(out/'history.json',history)
        save_checkpoint(out/'last.pt',model,optimizer,config=cfg,provenance=provenance,epoch=ep,step=steps,metrics=metrics,complete=False)
        if improved:save_checkpoint(out/'best_val.pt',model,optimizer,config=cfg,provenance=provenance,epoch=ep,step=steps,metrics=metrics)
        print(json.dumps(dict(method=cfg.get('method','original'),**row)),flush=True)
        if bad>=10:break
    save_checkpoint(out/'last.pt',model,optimizer,config=cfg,provenance=provenance,epoch=ep,step=steps,metrics=metrics,complete=True);model.load_state_dict(best_state)
    torch.save(dict(model=model.state_dict(),config=cfg,provenance=provenance,best_epoch=best_ep,retained_original_ids=data.keep),out/'deployment.tmp');(out/'deployment.tmp').replace(out/'deployment.pt');atomic_json(out/'complete.json',dict(epochs=ep,best_epoch=best_ep,complete=True))


def family(raw,keep,deleted_global,cfg,directory,device,provenance,shared_original_directory=None):
    from run_user_mia import subset
    from run_unlearning import ROOT,graph,ODSR,compact,seed_everything,digest,sensitivity,evaluate,atomic_json
    from social_unlearning.native_guarded import GuardedNativeForgetRetain
    from social_unlearning.baselines import DA2Social
    from social_unlearning.scif_social import gradient_change,native_probe,conjugate_gradient
    local=subset(raw,keep);original_data=graph(local,np.arange(local.U),device);seed_everything(cfg['seed']);original=ODSR(original_data,cfg,device).to(device)
    fit(original,original_data,dict(cfg,method='original'),shared_original_directory or directory/'original',dict(provenance,deletion='original'),cfg['original_epochs'],device);yield 'original',original
    remaining=np.flatnonzero(~np.isin(keep,deleted_global));retained=graph(local,remaining,device);prov=dict(provenance,deletion=digest(deleted_global))
    for method in ['retrain','retain_finetune','guarded','da2','scif']:
        seed_everything(cfg['seed']);student=ODSR(retained,cfg,device).to(device) if method=='retrain' else compact(original,retained,cfg,device)
        method_cfg=dict(cfg,method=method);out=directory/method
        if method in ['retrain','retain_finetune']:
            fit(student,retained,method_cfg,out,prov,cfg['original_epochs'] if method=='retrain' else cfg['unlearn_epochs'],device)
        elif method=='guarded':
            frozen=json.loads((ROOT/'checkpoints/tuning_native_guarded/epinions/seed_42/86ed2df5c4396fc5/f6cb0d3773ed017f/frozen_configuration.json').read_text());hyper=frozen['parameters'];method_cfg.update(hyper)
            teacher=compact(original,retained,cfg,device);weights=sensitivity(original,teacher,retained,np.random.default_rng(cfg['seed']),device);del teacher;mix=hyper['weight_mix'];weights['rec']=1+mix*(weights['rec']-1);weights['anchor']=1+mix*(weights['anchor']-1)
            objective=GuardedNativeForgetRetain(original,student,retained,local,weights,cfg['seed'],forget_strength=hyper['forget_strength'],forget_cl_strength=.1);objective.forget_rec=1+mix*(objective.forget_rec-1);objective.forget_anchor=1+mix*(objective.forget_anchor-1)
            fit(student,retained,method_cfg,out,prov,5,device,objective,guard=True);del objective,weights
        elif method=='da2':
            method_cfg.update(lr=.0005,eta0=1.,gamma=1.,delta=1,lambda_stability=1.,difficulty_tau=1.)
            objective=DA2Social(original,student,retained,local,original_data,ROOT/'external/baseline_sources/DA2Unlearn',cfg['seed'],method_cfg)
            fit(student,retained,method_cfg,out,prov,cfg.get('da2_epochs',300),device,objective,epoch_end=objective.epoch_end);del objective
        else:
            out.mkdir(parents=True,exist_ok=True)
            if (out/'complete.json').exists():
                saved=torch.load(out/'deployment.pt',map_location=device,weights_only=False)
                if saved['provenance']!=prov:raise ValueError('SCIF shadow provenance mismatch')
                student.load_state_dict(saved['model'])
            else:
                initial=copy.deepcopy(student.state_dict());rhs,probe,meta=gradient_change(original,student,retained,local,cfg['seed']);best=-1;best_state=None;trials=[]
                for damping in [.01,.1,1.]:
                    student.load_state_dict(initial);parameter=student.embedding_dict['user_emb'];loss=native_probe(student,*probe);g=torch.autograd.grad(loss,parameter,create_graph=True)[0]
                    def operator(v):return torch.autograd.grad((g*v).sum(),parameter,retain_graph=True)[0].detach()+damping*v
                    delta,info=conjugate_gradient(operator,rhs);record=dict(damping=damping,solver=info)
                    if info['status'] in ['converged','zero_rhs']:
                        norm=float(delta.norm())
                        with torch.no_grad():parameter.sub_(delta*min(1.,1/max(norm,1e-12)))
                        metrics,_=evaluate(student,retained,retained.val,device);record['validation']=metrics
                        if metrics['recall@20']>best:best=metrics['recall@20'];best_state=copy.deepcopy(student.state_dict())
                    trials.append(record);del g,loss,delta
                if best_state is None:raise RuntimeError('All shadow SCIF solves failed')
                student.load_state_dict(best_state);torch.save(dict(model=student.state_dict(),config=method_cfg,provenance=prov),out/'deployment.tmp');(out/'deployment.tmp').replace(out/'deployment.pt');save_checkpoint(out/'complete.pt',student,None,config=method_cfg,provenance=prov,epoch=1,step=1,complete=True);atomic_json(out/'complete.json',dict(probe=meta,trials=trials,complete=True))
        yield method,student
        del student
