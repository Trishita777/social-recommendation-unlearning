"""Exact-checkpoint DiffNet++ history MIA; same protocol as completed ODSR."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from run_unlearning import ROOT, atomic_json
from social_unlearning.diffnet_plus import DiffNetPlus, make_graph
from social_unlearning.history_mia import history_plan, history_features
from social_unlearning.interaction_mia import metrics, paired_bootstrap


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(base,raw,deleted,device):
    report=json.loads((base/'results.json').read_text());seed=report['seed']
    out=base/'mia';out.mkdir(exist_ok=True)
    plan=history_plan(raw,deleted,seed)
    ntrain=len(plan['roles']['train']);nval=len(plan['roles']['validation']);target=slice(ntrain+nval,None)
    np.savez_compressed(out/'sampling.npz',**{k:plan[k] for k in ['users','items','negatives','popularity_bins','history_items','history_indptr']})
    source_files={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'src/social_unlearning/history_mia.py',ROOT/'src/social_unlearning/interaction_mia.py',ROOT/'src/social_unlearning/diffnet_plus.py']}
    manifest=dict(protocol='shared_item_history_frozen_original_v1',source_files=source_files,roles=plan['roles'],
                  requested_users=plan['requested_users'],eligible_requested_users=plan['eligible_requested_users'],
                  provenance=report['provenance'],sampling_sha256=sha(out/'sampling.npz'),
                  interpretation='Interaction membership through shared item representations and supplied history excluding both audited items; not native user scoring or a user-membership guarantee.')
    atomic_json(out/'manifest.json',manifest)
    all_features={};hashes={}
    for method,row in report['methods'].items():
        path=Path(row['checkpoint']);hashes[method]=sha(path)
        ck=torch.load(path,map_location='cpu',weights_only=False)
        expected=dict(report['provenance'],deletion='original') if method=='original' else report['provenance']
        if ck['provenance']!=expected:raise ValueError('MIA checkpoint provenance mismatch')
        data=make_graph(raw,np.asarray(ck['retained_original_ids']))
        model=DiffNetPlus(data,ck['config']['dimension'],device);model.load_state_dict(ck['model']);model.eval()
        with torch.no_grad():_,items=model();x,_=history_features(items.cpu().numpy(),plan)
        all_features[method]=x
        del model,items,ck
    x=all_features['original'];train=x[:ntrain].reshape(-1,8);val=x[ntrain:ntrain+nval].reshape(-1,8)
    ytrain=np.tile([1,0],ntrain);yval=np.tile([1,0],nval);best=-float('inf');trials=[]
    candidates=[(f'logistic_C={c}',make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=1000,random_state=seed))) for c in [.01,.1,1.,10.]]
    candidates.append(('hist_gradient_boosting',HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=7,l2_regularization=1.,early_stopping=False,random_state=seed)))
    for name,clf in candidates:
        clf.fit(train,ytrain);prob=clf.predict_proba(val)[:,1];auc=float(roc_auc_score(yval,prob));trials.append(dict(name=name,validation_auc=auc))
        if auc>best:best=auc;attacker=clf;selected=name;selected_prob=prob
    threshold=float(np.nextafter(np.quantile(selected_prob[yval==0],.95,method='higher'),np.inf))
    joblib.dump(attacker,out/'attacker.joblib')
    fit=dict(name=selected,trials=trials,validation_auc=best,threshold=threshold,validation=metrics(selected_prob.reshape(-1,2),threshold),
             original_checkpoint_sha256=hashes['original'],fitting_users=ntrain,validation_users=nval)
    atomic_json(out/'attacker.json',fit)
    for method,x in all_features.items():
        prob=attacker.predict_proba(x[target].reshape(-1,8))[:,1].reshape(-1,2)
        result=dict(method=method,backbone='DiffNet++',dataset=report['dataset'],seed=seed,metrics=metrics(prob,threshold),
                    uncertainty=paired_bootstrap(prob,seed),checkpoint_sha256=hashes[method],original_checkpoint_sha256=hashes['original'],
                    attacker=fit,attacker_sha256=sha(out/'attacker.joblib'),manifest=manifest)
        atomic_json(out/(method+'.json'),result)
        np.savez_compressed(out/(method+'_audit.npz'),target_features=x[target],target_probabilities=prob)
        report['methods'][method]['mia']=result
        print(json.dumps(dict(method=method,mia_auc=result['metrics']['auc'],users=len(prob))),flush=True)
    report['status']='complete_utility_and_mia';atomic_json(base/'results.json',report)
    manifest=json.loads((base/'manifest.json').read_text());manifest['status']='complete_utility_and_mia';atomic_json(base/'manifest.json',manifest)
