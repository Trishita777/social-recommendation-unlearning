"""Separate controlled user-MIA experiment with a declared inductive history probe.
This experiment reserves true nonmembers, so its targets are trained separately
from the full-user utility matrix. Never attach these attack numbers to those
full-user checkpoints. No user embedding or user-ID lookup is used in probes.
"""
import argparse,copy,hashlib,json,time
from collections import defaultdict
from types import SimpleNamespace
import numpy as np
import torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,balanced_accuracy_score,roc_curve
from run_unlearning import ROOT,load_data,graph,ODSR,compact,sensitivity,train,seed_everything,digest,atomic_json


def subset(raw,keep):
    mapping=np.full(raw.U,-1,dtype=np.int64); mapping[keep]=np.arange(len(keep))
    def pairs(p):
        p=p[mapping[p[:,0]]>=0].copy(); p[:,0]=mapping[p[:,0]]; return p
    social=raw.social[(mapping[raw.social[:,0]]>=0)&(mapping[raw.social[:,1]]>=0)]
    return SimpleNamespace(U=len(keep),I=raw.I,train=pairs(raw.train),val=pairs(raw.val),test=pairs(raw.test),social=mapping[social])


def matched_queries(raw,members,nonmembers,rng):
    activity=np.bincount(raw.train[:,0],minlength=raw.U); degree=np.bincount(raw.social.ravel(),minlength=raw.U)
    def stratum(u): return (int(np.log2(max(1,activity[u]))),int(np.log2(max(1,degree[u]))))
    buckets=defaultdict(list)
    for u in rng.permutation(nonmembers):
        if activity[u]>=4:buckets[stratum(u)].append(int(u))
    yes=[]; no=[]
    for u in rng.permutation(members):
        bucket=buckets[stratum(u)]
        if activity[u]>=4 and bucket: yes.append(int(u)); no.append(bucket.pop())
    return np.array(yes+no,dtype=np.int64),np.array([1]*len(yes)+[0]*len(no),dtype=np.int64)


@torch.no_grad()
def history_features(model,raw,users,device):
    """Mean-item-vector history probe: identical interface for removed IDs/nonmembers.
    Split training histories deterministically in half. Form a normalized query
    from conditioning item vectors; score held-out history items and 256 fixed
    negatives. Features aggregate scores/ranks without looking up queried users.
    This is an inductive attacker, distinct from native transductive scoring.
    """
    model.eval(); _,items=model(); by=defaultdict(list)
    for u,i in raw.train: by[int(u)].append(int(i))
    rows=[]
    for u in users:
        rng=np.random.default_rng(700000+int(u)); history=np.array(sorted(by[int(u)])); rng.shuffle(history); split=max(1,len(history)//2)
        context=history[:split]; probe=history[split:]; positives=set(history.tolist()); negatives=[]
        while len(negatives)<256:
            n=int(rng.integers(raw.I))
            if n not in positives:negatives.append(n)
        query=torch.nn.functional.normalize(items[context].mean(0),dim=0)
        pos=(items[probe]@query).cpu().numpy(); neg=(items[negatives]@query).cpu().numpy(); ranks=(pos[:,None]>neg[None,:]).mean(1)
        rows.append([pos.mean(),pos.std(),np.quantile(pos,.1),np.quantile(pos,.9),neg.mean(),neg.std(),pos.mean()-neg.mean(),ranks.mean(),ranks.std(),float(np.log1p(np.exp(np.clip(neg.mean()-pos,-40,40))).mean())])
    return np.asarray(rows,dtype=np.float64)


def model_family(raw,keep,deleted_global,cfg,directory,device,provenance):
    local=subset(raw,keep); data=graph(local,np.arange(local.U),device); seed_everything(cfg['seed']); model=ODSR(data,cfg,device).to(device)
    train(model,data,cfg,directory/'original',dict(provenance,deletion='original'),cfg['original_epochs'],'original',device)
    yield 'original',model
    drop=np.flatnonzero(np.isin(keep,deleted_global)); remaining=np.setdiff1d(np.arange(local.U),drop); retained=graph(local,remaining,device); prov=dict(provenance,deletion=digest(deleted_global))
    teacher=compact(model,retained,cfg,device); weights=sensitivity(model,teacher,retained,np.random.default_rng(cfg['seed']),device)
    for method in ('retrain','retain_finetune','proposed'):
        seed_everything(cfg['seed']); student=ODSR(retained,cfg,device).to(device) if method=='retrain' else compact(model,retained,cfg,device)
        train(student,retained,cfg,directory/method,prov,cfg['original_epochs'] if method=='retrain' else cfg['unlearn_epochs'],method,device,weights if method=='proposed' else None)
        yield method,student
        del student


def attack_metrics(y,score,threshold,seed):
    pred=score>=threshold; auc=float(roc_auc_score(y,score)); balanced=float(balanced_accuracy_score(y,pred)); rng=np.random.default_rng(seed); boot=[]
    # Every feature row here is one distinct target user; resample users, not edges.
    for _ in range(1000):
        idx=rng.integers(len(y),size=len(y))
        if len(np.unique(y[idx]))==2: boot.append(roc_auc_score(y[idx],score[idx]))
    result=dict(roc_auc=auc,balanced_accuracy=balanced,auc_95ci=np.quantile(boot,[.025,.975]).tolist(),members=int(y.sum()),nonmembers=int((1-y).sum()),threshold=float(threshold),tpr_at_1pct_fpr=None)
    if (1-y).sum()>=1000:
        fpr,tpr,_=roc_curve(y,score); result['tpr_at_1pct_fpr']=float(tpr[fpr<=.01].max())
    else:result['low_fpr_note']='Not estimated: fewer than 1000 nonmembers.'
    return result


def run(a):
    torch.set_num_threads(8); device=torch.device(a.device); raw=load_data(a.dataset,a.seed); directory=ROOT/'results/user_mia'/a.dataset/f'seed_{a.seed}'; directory.mkdir(parents=True,exist_ok=True)
    base=ROOT/'checkpoints/ODSR'/a.dataset/f'seed_{a.seed}'/raw.meta['split_hash'][:16]; utility=torch.load(base/'original/deployment.pt',map_location='cpu',weights_only=False); cfg=utility['config']; cfg=dict(cfg,inductive_mia=True)
    ids=np.random.default_rng(9001).permutation(raw.U); n=raw.U
    attack_train=ids[:int(.4*n)]; attack_val=ids[int(.4*n):int(.6*n)]; target_ids=ids[int(.6*n):int(.8*n)]; context=ids[int(.8*n):]
    assert not set(attack_train)&set(attack_val) and not set(attack_train)&set(target_ids) and not set(attack_val)&set(target_ids)
    source=hashlib.sha256(open(__file__,'rb').read()).hexdigest()
    provenance=dict(split=raw.meta['split_hash'],source=source,partition=digest(ids),dataset=hashlib.sha256(json.dumps(raw.meta['source_files'],sort_keys=True).encode()).hexdigest())
    manifest=dict(protocol='separate controlled target with true nonmembers; cannot be reported as attack on full-user utility checkpoints',feature_schema=['positive_mean','positive_std','positive_q10','positive_q90','negative_mean','negative_std','margin','relative_rank_mean','relative_rank_std','softplus_pair_loss'],probe='normalized mean of conditioning item vectors; no queried user embedding lookup',attack_train_users=attack_train.tolist(),attack_validation_users=attack_val.tolist(),target_evaluation_users=target_ids.tolist(),context_users=context.tolist(),shadow_seeds=[1042,1043,1044,1045],config=cfg,provenance=provenance)
    atomic_json(directory/'manifest.json',manifest)
    collected=defaultdict(list)
    for seed in [1042,1043,1044,1045,a.seed]:
        target=seed==a.seed; rng=np.random.default_rng(seed); candidates=target_ids if target else np.concatenate([attack_train,attack_val]); shuffled=rng.permutation(candidates); member=shuffled[:len(shuffled)//2]; nonmember=shuffled[len(shuffled)//2:]
        background=np.concatenate([context,attack_train,attack_val]) if target else context
        keep=np.sort(np.concatenate([background,member])); deleted=rng.choice(member,max(1,round(len(keep)*a.fraction)),replace=False)
        queries,labels=matched_queries(raw,deleted,nonmember,rng)
        if len(queries)<20:raise ValueError('Too few matched user examples')
        roles=np.array(['test' if target else ('train' if u in set(attack_train) else 'validation') for u in queries])
        modeldir=ROOT/'checkpoints/shadow_mia'/a.dataset/f'seed_{seed}'/digest(keep)[:16]
        atomic_json(modeldir/'membership.json',dict(keep=keep.tolist(),deleted=deleted.tolist(),queries=queries.tolist(),labels=labels.tolist(),roles=roles.tolist(),target=target))
        for method,model in model_family(raw,keep,deleted,dict(cfg,seed=seed),modeldir,device,dict(provenance,membership=digest(keep))):
            features=history_features(model,raw,queries,device)
            for role in ('train','validation','test'):
                mask=roles==role
                if mask.any(): collected[(method,role)].append((features[mask],labels[mask],queries[mask]))
            np.savez(directory/f'features_seed{seed}_{method}.npz',features=features,labels=labels,users=queries,roles=roles)
            print('MIA_FEATURES',seed,method,len(queries),flush=True)
    def data(method,role):
        triples=collected[(method,role)]; return tuple(np.concatenate([t[k] for t in triples]) for k in range(3))
    import joblib
    results=[]
    for method in ('original','retrain','retain_finetune','proposed'):
        xt,yt,ut=data(method,'test')
        for adaptivity in ('frozen','adaptive'):
            shadow_method='original' if adaptivity=='frozen' else method
            x,y,u=data(shadow_method,'train'); xv,yv,uv=data(shadow_method,'validation')
            assert not set(u)&set(uv) and not set(u)&set(ut) and not set(uv)&set(ut)
            for kind in ('logistic','mlp'):
                candidates=[]
                for regularizer in (.01,1,100):
                    estimator=LogisticRegression(C=regularizer,max_iter=1000,random_state=1) if kind=='logistic' else MLPClassifier(hidden_layer_sizes=(32,),alpha=1/regularizer,max_iter=500,random_state=1,early_stopping=False)
                    pipe=make_pipeline(StandardScaler(),estimator); pipe.fit(x,y); scores=pipe.predict_proba(xv)[:,1]; candidates.append((roc_auc_score(yv,scores),pipe,scores,regularizer))
                val_auc,pipe,scores,hyper=max(candidates,key=lambda z:z[0]); fpr,tpr,thresholds=roc_curve(yv,scores); threshold=thresholds[np.argmax(tpr-fpr)]
                metrics=attack_metrics(yt,pipe.predict_proba(xt)[:,1],threshold,a.seed)
                joblib.dump(pipe,directory/f'attacker_{adaptivity}_{kind}_{method}.joblib')
                row=dict(target=method,attacker=kind,adaptivity=adaptivity,shadow_validation_auc=float(val_auc),selected_regularizer=hyper,attack_training_rows=len(y),attack_training_unique_users=len(set(u)),attack_validation_unique_users=len(set(uv)),**metrics); results.append(row)
    atomic_json(directory/'results.json',dict(status='complete_separate_controlled_mia_experiment',target_scope=manifest['protocol'],rows=results,interpretation='Compare original attack strength and scratch references; chance original performance is inconclusive, not proof of forgetting.'))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--dataset',default='epinions'); p.add_argument('--seed',type=int,default=42); p.add_argument('--fraction',type=float,default=.01); p.add_argument('--device',default='cuda:1'); run(p.parse_args())
