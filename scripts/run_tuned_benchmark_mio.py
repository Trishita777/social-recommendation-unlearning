"""Separate shadow-validation-tuned MIO diagnostic, preserving benchmark features.

Does not replace the default SGD(.001,100 epochs) audit. No target inputs enter
fitting or selection. Reuses frozen recommender checkpoints for evaluation.
"""
import argparse
import copy
import fcntl
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,roc_curve
from run_unlearning import ROOT,ODSR,load_data,atomic_json
from run_egcl_backbone import atomic_torch
from social_unlearning.edge_deletion import edge_graph,filtered_raw
from social_unlearning.benchmark_mio import BenchmarkMIO,embedding_features
from social_unlearning.controlled_mio import attack_metrics,bootstrap_attack
from run_controlled_target_mio import make_graphs,load_model

GRID=[dict(optimizer='SGD',lr=.01,epochs=300),dict(optimizer='Adam',lr=.001,epochs=100)]


def fit_candidate(x,y,vx,vy,cfg):
    torch.manual_seed(9042);model=BenchmarkMIO(x.shape[1])
    optimizer=getattr(torch.optim,cfg['optimizer'])(model.parameters(),lr=cfg['lr'])
    generator=torch.Generator().manual_seed(9042);best=float('inf');bad=0;history=[]
    for epoch in range(1,cfg['epochs']+1):
        model.train()
        for ids in torch.randperm(len(x),generator=generator).split(512):
            optimizer.zero_grad(set_to_none=True)
            loss=torch.nn.functional.cross_entropy(model(x[ids]),y[ids]);loss.backward();optimizer.step()
        model.eval()
        with torch.no_grad():value=float(torch.nn.functional.cross_entropy(model(vx),vy))
        if not np.isfinite(value):raise FloatingPointError('Nonfinite validation loss')
        history.append(value)
        if value<best:
            best=value;state=copy.deepcopy(model.state_dict());best_epoch=epoch;bad=0
        else:bad+=1
        if bad>=20:break
    model.load_state_dict(state);model.eval()
    return model,dict(config=cfg,best_validation_loss=best,best_epoch=best_epoch,epochs=epoch,history=history)


def histories(raw):
    result=defaultdict(list)
    for u,i in np.concatenate([raw.train,raw.val]):result[int(u)].append(int(i))
    return result


def run(a):
    torch.set_num_threads(4);device=torch.device(a.device)
    bank=a.bank.resolve();manifest=json.loads((bank/'manifest.json').read_text())
    source=hashlib.sha256(Path(__file__).read_bytes()+(ROOT/'src/social_unlearning/benchmark_mio.py').read_bytes()).hexdigest()
    out=ROOT/'checkpoints/tuned_benchmark_mio/ODSR'/a.dataset/source[:16];out.mkdir(parents=True,exist_ok=True)
    with (out/'run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not (out/'attacker.pt').exists():
            raw=load_data(a.dataset,42);members=np.array(manifest['shadow_member_users'])
            assert manifest['provenance']['split']==raw.meta['split_hash']
            ck=torch.load(bank/'shadow_original/deployment.pt',map_location='cpu',weights_only=False)
            data=edge_graph(raw,np.setdiff1d(np.arange(raw.U),members),device)
            model=ODSR(data,ck['config'],device).to(device);model.load_state_dict(ck['model']);model.eval()
            with torch.no_grad():users,items=model();users=users.cpu();items=items.cpu()
            del model,data
            known=histories(raw);features={};labels={}
            for role,ids in manifest['roles'].items():
                features[role]=embedding_features(users,items,ids,{u:u for u in range(raw.U)},known)
                labels[role]=torch.tensor(np.isin(ids,members).astype(np.int64))
            mean=features['train'].mean(0);std=features['train'].std(0,unbiased=False).clamp_min(1e-6)
            features={k:(x-mean)/std for k,x in features.items()}
            reports=[];selected=None;best=float('inf')
            for cfg in GRID:
                candidate,report=fit_candidate(features['train'],labels['train'],features['validation'],labels['validation'],cfg)
                reports.append(report);print('CANDIDATE',cfg,report['best_validation_loss'],flush=True)
                if report['best_validation_loss']<best:
                    best=report['best_validation_loss'];selected=copy.deepcopy(candidate.state_dict());selected_cfg=cfg
            attacker=BenchmarkMIO(features['train'].shape[1]);attacker.load_state_dict(selected)
            val=attacker.member_probability(features['validation']).numpy()
            fpr,tpr,thresholds=roc_curve(labels['validation'].numpy(),val);threshold=float(thresholds[np.argmax(tpr-fpr)])
            test=attacker.member_probability(features['test']).numpy()
            validation=dict(status='separate_validation_tuned_adaptation',default_attacker=str(bank/'attacker.pt'),
                source=source,grid=GRID,selected=selected_cfg,candidates=reports,
                selection='minimum shadow-validation cross-entropy; threshold maximizes shadow-validation TPR-FPR',
                normalization='per-coordinate mean/std from shadow train only; NOT unit-length per-user normalization',
                shadow_test=attack_metrics(labels['test'].numpy(),test,threshold),
                threat_model='native user plus mean known-history item embeddings; no target inputs used in tuning')
            atomic_json(out/'validation.json',validation)
            atomic_torch(out/'attacker.pt',dict(model=selected,mean=mean,std=std,threshold=threshold,
                original_shadow_manifest=hashlib.sha256((bank/'manifest.json').read_bytes()).hexdigest(),source=source))
        saved=torch.load(out/'attacker.pt',map_location='cpu',weights_only=False)
        if saved['original_shadow_manifest']!=hashlib.sha256((bank/'manifest.json').read_bytes()).hexdigest():raise ValueError('Shadow manifest changed')
        attacker=BenchmarkMIO(len(saved['mean']));attacker.load_state_dict(saved['model'])
        for base in sorted((ROOT/'checkpoints/controlled_target_mio/ODSR'/a.dataset).glob('seed_*/*')):
            if not (base/'manifest.json').exists():continue
            protocol=json.loads((base/'manifest.json').read_text())
            if protocol['provenance']['shadow_manifest']!=saved['original_shadow_manifest']:continue
            seed=int(base.parent.name.removeprefix('seed_'));raw=load_data(a.dataset,seed)
            deleted=np.array(protocol['requested_users']);nonmembers=np.array(protocol['never_member_users'])
            queries=np.concatenate([deleted,nonmembers]);labels=np.r_[np.ones(len(deleted),dtype=int),np.zeros(len(nonmembers),dtype=int)]
            forbidden=np.concatenate([np.array(ids) for ids in manifest['roles'].values()])
            assert not np.intersect1d(queries,forbidden).size
            before,after=make_graphs(raw,deleted,nonmembers,device);raw_members=filtered_raw(raw,nonmembers)
            known=histories(raw);rows=[]
            for completion in sorted(base.glob('*/completion.json')):
                method=completion.parent.name
                model,ck=load_model(completion.parent,before if method=='original' else after,raw_members,deleted,device)
                assert ck['provenance']==dict(protocol['provenance'],method=method)
                with torch.no_grad():
                    users,items=model();x=embedding_features(users.cpu(),items.cpu(),queries,{u:u for u in range(raw.U)},known)
                probability=attacker.member_probability((x-saved['mean'])/saved['std']).numpy()
                rows.append(dict(method=method,metrics=attack_metrics(labels,probability,saved['threshold']),
                    uncertainty=bootstrap_attack(labels,probability,saved['threshold'],seed=seed),probabilities=probability.tolist()))
                del model,users,items,x
            report=dict(status='controlled_target_tuned_mio_diagnostic',dataset=a.dataset,seed=seed,
                target_protocol=protocol['provenance'],attacker=str(out/'attacker.pt'),rows=rows,
                limitation='Additional tuned attack; benchmark-default results retained separately. Check original target discrimination; chance alone is not proof of unlearning.')
            destination=ROOT/'results/controlled_target_tuned_mio/ODSR'/a.dataset/base.parent.name/base.name/source[:16]/'results.json'
            atomic_json(destination,report);print('TUNED_TARGET_RESULT',destination,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--bank',type=Path,required=True)
    p.add_argument('--device',default='cuda:0');run(p.parse_args())
