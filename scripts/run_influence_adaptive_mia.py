"""Supplementary method-adaptive history-query audit with user-disjoint selection.

Primary frozen-bank scores are preserved. Fit each model's attacker only on
retained fitting users, select only on retained validation, and evaluate the
same untouched requested interactions. No requested outcomes choose a head.
"""
import hashlib,json,os,time,fcntl
from pathlib import Path
import joblib,numpy as np,torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from run_unlearning import ROOT,load_data,atomic_json
from run_influence_mia import extract,history_plan,sha
from social_unlearning.interaction_mia import metrics,paired_bootstrap


def audit(bb,ds,seed):
 raw=load_data(ds,seed);deleted=np.sort(np.random.default_rng(seed+10000).permutation(raw.U)[:max(1,round(raw.U*.01))]);plan=history_plan(raw,deleted,seed)
 found=list((ROOT/'checkpoints/interaction_mia_history_influence_v4'/bb/ds/f'seed_{seed}').glob('*/sorecremover_influence.json'))
 if not found or not (found[0].parent/'graph_only.json').exists():return False
 assert len(found)==1
 bank=found[0].parent;out=ROOT/'checkpoints/interaction_mia_history_adaptive_v3'/bb/ds/f'seed_{seed}'/bank.name;out.mkdir(parents=True,exist_ok=True)
 ntrain=len(plan['roles']['train']);nval=len(plan['roles']['validation']);ytrain=np.tile([1,0],ntrain);yval=np.tile([1,0],nval)
 for method in ['original','retrain','graph_only','sorecremover_influence']:
  dest=out/(method+'.json')
  if dest.exists():continue
  base=json.loads((bank/(method+'.json')).read_text());checkpoint_sha=sha(base['checkpoint']);assert checkpoint_sha==base['checkpoint_sha256']
  if method=='original':
   xx=np.load(bank/'original_features.npz')['x'];diag=base['diagnostics']
  else:
   source=ROOT/base['utility_source_file'];utility=json.loads(source.read_text())
   row=dict(method=method,checkpoint=base['checkpoint'],graph=base['inference_graph'],provenance=utility['provenance'])
   xx,diag=extract(row,raw,deleted,bb,plan,torch.device('cpu'))
  train=xx[:ntrain].reshape(-1,8);val=xx[ntrain:ntrain+nval].reshape(-1,8)
  candidates=[(f'logistic_C={c}',make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=1000,random_state=seed))) for c in [.01,.1,1.,10.]]
  candidates.append(('hist_gradient_boosting',HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=7,l2_regularization=1.,early_stopping=False,random_state=seed)))
  trials=[];best=-float('inf')
  for name,clf in candidates:
   clf.fit(train,ytrain);p=clf.predict_proba(val)[:,1];auc=float(roc_auc_score(yval,p));trials.append(dict(name=name,validation_auc=auc))
   if auc>best:best=auc;selected=clf;selected_name=name;val_prob=p
  threshold=float(np.nextafter(np.quantile(val_prob[yval==0],.95,method='higher'),np.inf))
  joblib.dump(selected,out/(method+'_attacker.joblib'))
  prob=selected.predict_proba(xx[ntrain+nval:].reshape(-1,8))[:,1].reshape(-1,2)
  np.savez_compressed(out/(method+'_audit.npz'),target_features=xx[ntrain+nval:],target_probabilities=prob)
  hashes=dict(base['source_hashes']);hashes[str(Path(__file__).relative_to(ROOT))]=sha(__file__)
  report=dict(base,protocol='interaction_mia_history_adaptive_v3',metrics=metrics(prob,threshold),uncertainty=paired_bootstrap(prob,seed),diagnostics=diag,
              attacker=dict(name=selected_name,trials=trials,validation_auc=best,threshold=threshold,selection='method-specific retained-user validation; requested users excluded'),
              attacker_sha256=sha(out/(method+'_attacker.joblib')),source_hashes=hashes,
              sampling_manifest=str(bank/'sampling.npz'),frozen_bank_report=str(bank/(method+'.json')),
              interpretation='Supplementary method-adaptive history-probe interaction audit; shared item representations, no native deleted-user query. Head and threshold selected only on disjoint retained validation users. Not certified erasure.')
  atomic_json(dest,report);print(bb,ds,seed,method,report['metrics']['auc'],flush=True)
 return True

if __name__=='__main__':
 torch.set_num_threads(4);base=ROOT/'results/influence_adaptive_mia_v3_20261112';base.mkdir(parents=True,exist_ok=True)
 lock=(base/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 while True:
  complete=0
  for bb in ['ODSR','EGCL']:
   for ds in ['epinions','yelp2018']:
    for seed in [42,43,44]:
     atomic_json(base/'status.json',dict(state='running',active=f'{bb}_{ds}_{seed}',completed_cases=complete))
     complete+=audit(bb,ds,seed)
  atomic_json(base/'status.json',dict(state='complete' if complete==12 else 'waiting_for_new_checkpoints',active=None,completed_cases=complete))
  if complete==12:break
  time.sleep(20)
