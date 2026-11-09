"""CURE4Rec-style native-representation audit, one reusable full-ID shadow bank.
Targets are the actual utility checkpoints. All requested target users were
original members: report positive-only membership probabilities/rates and gaps
to retraining, NOT target AUC or balanced accuracy without genuine nonmembers.
"""
import argparse,hashlib,json,fcntl
from collections import defaultdict
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,balanced_accuracy_score
from run_unlearning import ROOT,load_data,seed_everything,ODSR,train,atomic_json,digest
from run_egcl_backbone import EGCL,train as train_egcl,load_parameters,parameter_state,atomic_torch
from social_unlearning.edge_deletion import edge_graph
from social_unlearning.benchmark_mio import BenchmarkMIO,embedding_features,train_mio
from audit_corrected_forgetting import representations,paths


def interval(values,seed=1):
    values=np.asarray(values);rng=np.random.default_rng(seed)
    boot=[values[rng.integers(len(values),size=len(values))].mean() for _ in range(1000)]
    return np.quantile(boot,[.025,.975]).tolist()


def run(a):
    torch.set_num_threads(4);device=torch.device(a.device);torch.cuda.set_device(device)
    raw=load_data(a.dataset,42)
    path=ROOT/'checkpoints'/a.backbone/a.dataset/'seed_42'/raw.meta['split_hash'][:16]/'original/deployment.pt'
    original=torch.load(path,map_location='cpu',weights_only=False)
    cfg=dict(original['config'],seed=1042,mio_shadow=True)
    target_union=np.unique(np.concatenate([np.random.default_rng(seed+10000).permutation(raw.U)[:max(1,round(raw.U*.01))] for seed in [42,43,44]]))
    eligible=np.setdiff1d(np.arange(raw.U),target_union);rng=np.random.default_rng(9042);eligible=rng.permutation(eligible)
    roles=dict(train=eligible[:int(.6*len(eligible))],validation=eligible[int(.6*len(eligible)):int(.8*len(eligible))],test=eligible[int(.8*len(eligible)):])
    members=np.sort(np.concatenate([rng.permutation(users)[:len(users)//2] for users in roles.values()]))
    absent=np.setdiff1d(np.arange(raw.U),members)
    assert not np.intersect1d(target_union,members).size
    from pathlib import Path
    source=hashlib.sha256(b''.join(p.read_bytes() for p in [Path(__file__),ROOT/'src/social_unlearning/benchmark_mio.py',ROOT/'src/social_unlearning/edge_deletion.py',ROOT/'scripts/audit_corrected_forgetting.py',ROOT/'scripts/run_unlearning.py',ROOT/'src/social_unlearning/egcl.py',ROOT/'scripts/run_egcl_backbone.py'])).hexdigest()
    provenance=dict(source=source,split=raw.meta['split_hash'],membership=digest(members),backbone=a.backbone,deletion_semantics='incident_edges_only_keep_user_rows',purpose='one_original_shadow_model_for_benchmark_MIO')
    bank=ROOT/'checkpoints/benchmark_mio'/a.backbone/a.dataset/source[:16];bank.mkdir(parents=True,exist_ok=True)
    with (bank/'run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        out=bank/'shadow_original';out.mkdir(exist_ok=True)
        data=edge_graph(raw,absent,device);seed_everything(1042)
        shadow=ODSR(data,cfg,device).to(device) if a.backbone=='ODSR' else EGCL(data,cfg,device)
        if (out/'deployment.pt').exists():
            ck=torch.load(out/'deployment.pt',map_location='cpu',weights_only=False)
            if ck['provenance']!=provenance:raise ValueError('Shadow provenance mismatch')
            if a.backbone=='ODSR':shadow.load_state_dict(ck['model'])
            else:load_parameters(shadow,ck['model'])
        elif a.backbone=='ODSR':
            timing=train(shadow,data,cfg,out,provenance,cfg['original_epochs'],'original',device)
            atomic_json(out/'timing.json',timing)
        else:
            timing=train_egcl(shadow,data,cfg,provenance,out,device);atomic_json(out/'timing.json',timing)
        shadow.eval()
        with torch.no_grad():u,i=shadow();u=u.cpu();i=i.cpu()
        del shadow,data
        histories=defaultdict(list)
        for user,item in np.concatenate([raw.train,raw.val]):histories[int(user)].append(int(item))
        # Historical items are attacker-known inputs, not added to any model graph.
        mapping={u:u for u in range(raw.U)};features={};labels={}
        for role,users in roles.items():
            features[role]=embedding_features(u,i,users,mapping,histories)
            labels[role]=torch.as_tensor(np.isin(users,members).astype(np.int64))
        del u,i
        atomic_json(bank/'manifest.json',dict(provenance=provenance,config=cfg,shadow_member_users=members.tolist(),
            roles={k:v.tolist() for k,v in roles.items()},excluded_target_users=target_union.tolist(),
            feature_schema='concat(native_propagated_user_embedding, mean_native_item_embedding_of_known_train_plus_validation_history)',
            target_scope='actual full-population utility checkpoints; positive-only requested-user audit',
            limitation='No genuine nonmember users in full-population target; no target AUC/balanced-accuracy is reported. Shadow test AUC is attack validation, not target privacy. One-shadow adaptation, not exact CURE4Rec code reproduction.'))
        if (bank/'attacker.pt').exists():
            saved=torch.load(bank/'attacker.pt',map_location='cpu',weights_only=False)
            attacker=BenchmarkMIO(features['train'].shape[1]);attacker.load_state_dict(saved['model']);attacker.eval();training=saved['training']
        else:
            attacker,training=train_mio(features['train'],labels['train'],features['validation'],labels['validation'])
            atomic_torch(bank/'attacker.pt',dict(model=attacker.state_dict(),training=training,provenance=provenance))
        with torch.no_grad():score=attacker.member_probability(features['test']).numpy()
        y=labels['test'].numpy()
        attack_validation=dict(shadow_test_auc=float(roc_auc_score(y,score)),shadow_test_balanced_accuracy=float(balanced_accuracy_score(y,score>=.5)),
            users=len(y),members=int(y.sum()),nonmembers=int((1-y).sum()),scope='held-out shadow users, not target users')
        atomic_json(bank/'attack_validation.json',attack_validation)
        for seed in [42,43,44]:
            target_raw=load_data(a.dataset,seed)
            if target_raw.um!=raw.um or target_raw.im!=raw.im:raise ValueError('User/item map drift')
            deleted=np.sort(np.random.default_rng(seed+10000).permutation(raw.U)[:max(1,round(raw.U*.01))])
            available=paths(a.backbone,a.dataset,seed,target_raw,deleted)
            if 'retrain' not in available:print('PENDING_RETRAIN',seed,flush=True);continue
            probabilities={};target_provenance={}
            for method,checkpoint in available.items():
                (users,items),prov=representations(a.backbone,method,target_raw,deleted,checkpoint,device)
                x=embedding_features(users,items,deleted,mapping,histories)
                probabilities[method]=attacker.member_probability(x).numpy();target_provenance[method]=prov
                del users,items,x
            rows=[]
            for method,prob in probabilities.items():
                gap=prob-probabilities['retrain']
                rows.append(dict(method=method,requested_users=len(prob),mean_membership_probability=float(prob.mean()),
                    membership_probability_95ci=interval(prob,seed),member_classification_rate=float((prob>=.5).mean()),
                    mean_probability_gap_to_retrain=float(gap.mean()),gap_95ci=interval(gap,seed),
                    target_auc=None,target_balanced_accuracy=None,provenance=target_provenance[method]))
            report=dict(status='completed_positive_only_benchmark_style_MIO',backbone=a.backbone,dataset=a.dataset,seed=seed,fraction=.01,
                attack_validation=attack_validation,attacker=str(bank/'attacker.pt'),training=training,
                rows=rows,interpretation='Read shadow attack strength and original requested-user detection first. Compare unlearned membership probabilities with matched retraining; smaller scores alone do not prove forgetting. This is not target AUC, accuracy or a privacy guarantee.')
            destination=ROOT/'results/benchmark_mio'/a.backbone/a.dataset/f'seed_{seed}'/'results.json'
            atomic_json(destination,report);print('BENCHMARK_MIO_RESULT',destination,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--backbone',choices=['ODSR','EGCL'],default='ODSR');p.add_argument('--dataset',required=True);p.add_argument('--device',default='cuda:0');run(p.parse_args())
