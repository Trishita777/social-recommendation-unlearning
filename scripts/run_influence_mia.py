"""Exact-checkpoint shared-item/history interaction MIA for compact deletion."""
import argparse, hashlib, json, gc, sys
from pathlib import Path
import numpy as np
import torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import joblib
from run_unlearning import ROOT, ODSR, load_data, graph, digest, atomic_json, seed_everything
from run_egcl_backbone import load_parameters
from run_egcl_unlearning_adapters import native_data
from social_unlearning.edge_deletion import edge_graph
from social_unlearning.egcl import EGCL
from social_unlearning.gnndelete_edges import GNNDeleteODSR, neighborhoods
from social_unlearning.egcl_unlearning_adapters import GNNDeleteEGCL
from social_unlearning.interaction_mia import metrics, paired_bootstrap
from social_unlearning.history_mia import history_plan,history_features
from run_interaction_mia import records_for as historical_records


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def records_for(bb, ds, seed, raw, deleted):
    rows=[r for r in historical_records(bb,ds,seed,raw,deleted) if not r['method'].startswith('sorecremover_dual')]
    for p in sorted((ROOT/'checkpoints/sorecremover_influence_spillover_v4'/bb/ds/f'seed_{seed}').rglob('results.json')):
        row=json.loads(p.read_text())
        if row['fraction']!=.01:continue
        rows.append(dict(method=row['method'],checkpoint=str(p.parent/'deployment.pt'),graph='compact_retained',source_file=str(p.relative_to(ROOT)),provenance=row['provenance']))
    if len({r['method'] for r in rows})!=len(rows):raise ValueError('Ambiguous current checkpoint versions')
    return rows


@torch.no_grad()
def extract(row,raw,deleted,bb,plan,device):
    ck=torch.load(row['checkpoint'],map_location='cpu',weights_only=False)
    assert ck['provenance']['split']==raw.meta['split_hash']
    assert ck['provenance']['social']==raw.meta['social_hash']
    if row['method']!='original':
        assert ck['provenance']['deletion']==digest(deleted)
        assert row['provenance']==ck['provenance']
    retained=row['graph']=='retained'
    compact=row['graph']=='compact_retained'
    data=(edge_graph(raw,deleted,device) if retained else graph(raw,np.arange(raw.U),device)) if bb=='ODSR' else native_data(raw,deleted if retained else None,device)
    if compact:
        data=graph(raw,np.asarray(ck['retained_original_ids']),device)
        if bb=='EGCL':del data.adj_ui,data.adj_uu,data.norm_adj
    cfg=ck['config'];model=ODSR(data,cfg,device).to(device) if bb=='ODSR' else EGCL(data,cfg,device)
    if row['method'] in ['gnndelete','gnndelete_native']:
        masks,_,_=neighborhoods(raw,deleted,cfg['n_layer'] if bb=='ODSR' else cfg['layer_size'])
        cls=GNNDeleteODSR if bb=='ODSR' else GNNDeleteEGCL
        model=cls(model,masks,ROOT/'external/baseline_sources/GNNDelete')
    if set(ck['model'])==set(dict(model.named_parameters())):load_parameters(model,ck['model'])
    else:model.load_state_dict(ck['model'],strict=True)
    model.eval();u,i=model();x,query=history_features(i.cpu().numpy(),plan)
    norms=np.linalg.norm(query[-len(plan['roles']['target']):],axis=1)

    del model,u,i,data,ck;gc.collect()
    return x,dict(history_query_max_norm=float(norms.max()),history_queries_zero_norm=int((norms==0).sum()))


def run(a):
    torch.set_num_threads(4);seed_everything(a.seed);device=torch.device(a.device)
    raw=load_data(a.dataset,a.seed)
    deleted=np.sort(np.random.default_rng(a.seed+10000).permutation(raw.U)[:max(1,round(raw.U*.01))])
    plan=history_plan(raw,deleted,a.seed)
    paths=[Path(__file__),ROOT/'scripts/run_interaction_mia.py',ROOT/'src/social_unlearning/history_mia.py',ROOT/'src/social_unlearning/interaction_mia.py',ROOT/'scripts/run_unlearning.py',ROOT/'scripts/run_egcl_backbone.py',ROOT/'scripts/run_egcl_unlearning_adapters.py',ROOT/'src/social_unlearning/egcl.py',ROOT/'src/social_unlearning/edge_deletion.py',ROOT/'src/social_unlearning/gnndelete_edges.py',ROOT/'src/social_unlearning/egcl_unlearning_adapters.py',ROOT/'external/ODSR/ODSR.py',ROOT/'external/baseline_sources/GNNDelete/framework/models/deletion.py']
    hashes={str(p.relative_to(ROOT)):sha(p) for p in paths}
    identity=dict(protocol='interaction_mia_history_influence_v4',source_files=hashes,split=raw.meta['split_hash'],social=raw.meta['social_hash'],data=raw.meta['source_files'],deletion=digest(deleted),
                  specification='shared-item history query excluding both audited items; same construction for every method; one train/test pair per user matched by floor(log2(1+original train item count)); user-disjoint attacker fit/validation/target; 64 fixed nonedges; 8 ID-free score/embedding features; frozen original-target attacker; validation-selected logistic or histogram gradient boosting; validation nonmember 95th percentile threshold',
                  attacker_access='original-model item representations plus supplied historical interactions and known interaction membership for retained users; no requested-user examples used in attacker fitting or selection')
    version=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16]
    out=ROOT/'checkpoints/interaction_mia_history_influence_v4'/a.backbone/a.dataset/f'seed_{a.seed}'/version;out.mkdir(parents=True,exist_ok=True)
    manifest=dict(identity,roles=plan['roles'],requested_users=plan['requested_users'],eligible_requested_users=plan['eligible_requested_users'],feature_names=['score','cosine','user_norm','item_norm','score_minus_negative_mean','standardized_score','negative_percentile','negative_std'])
    atomic_json(out/'manifest.json',manifest)
    np.savez_compressed(out/'sampling.npz',users=plan['users'],items=plan['items'],negatives=plan['negatives'],popularity_bins=plan['popularity_bins'],history_items=plan['history_items'],history_indptr=plan['history_indptr'])
    ntrain=len(plan['roles']['train']);nval=len(plan['roles']['validation']);target=slice(ntrain+nval,None)
    rows=records_for(a.backbone,a.dataset,a.seed,raw,deleted);original=rows[0]
    original_sha=sha(original['checkpoint'])
    original_features=out/'original_features.npz'
    if not original_features.exists():
        x,diag=extract(original,raw,deleted,a.backbone,plan,device)
        np.savez_compressed(original_features,x=x);atomic_json(out/'original_diagnostics.json',diag)
    x=np.load(original_features)['x']
    if not (out/'attacker.joblib').exists():
        train=x[:ntrain].reshape(-1,8);val=x[ntrain:ntrain+nval].reshape(-1,8)
        ytrain=np.tile([1,0],ntrain);yval=np.tile([1,0],nval);candidates=[]
        for c in [.01,.1,1.,10.]:candidates.append((f'logistic_C={c}',make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=1000,random_state=a.seed))))
        candidates.append(('hist_gradient_boosting',HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=7,l2_regularization=1.,early_stopping=False,random_state=a.seed)))
        trials=[];best=-float('inf')
        for name,clf in candidates:
            clf.fit(train,ytrain);prob=clf.predict_proba(val)[:,1];auc=float(roc_auc_score(yval,prob));trials.append(dict(name=name,validation_auc=auc))
            if auc>best:best=auc;selected=clf;selected_name=name;selected_p=prob
        threshold=float(np.nextafter(np.quantile(selected_p[yval==0],.95,method='higher'),np.inf))
        joblib.dump(selected,out/'attacker.joblib')
        atomic_json(out/'attacker.json',dict(name=selected_name,trials=trials,validation_auc=best,threshold=threshold,validation=metrics(selected_p.reshape(-1,2),threshold),original_checkpoint_sha256=original_sha))
    fit=json.loads((out/'attacker.json').read_text());assert fit['original_checkpoint_sha256']==original_sha
    attacker=joblib.load(out/'attacker.joblib');sampling_sha=sha(out/'sampling.npz')
    for row in rows:
        checkpoint_sha=sha(row['checkpoint']);dest=out/(row['method']+'.json')
        if dest.exists():
            previous=json.loads(dest.read_text());assert previous['checkpoint_sha256']==checkpoint_sha and previous['original_checkpoint_sha256']==original_sha
            continue
        if row['method']=='original':xx=x;diag=json.loads((out/'original_diagnostics.json').read_text())
        else:xx,diag=extract(row,raw,deleted,a.backbone,plan,device)
        prob=attacker.predict_proba(xx[target].reshape(-1,8))[:,1].reshape(-1,2)
        np.savez_compressed(out/(row['method']+'_audit.npz'),target_features=xx[target],target_probabilities=prob)
        report=dict(backbone=a.backbone,dataset=a.dataset,seed=a.seed,fraction=.01,method=row['method'],protocol=identity['protocol'],checkpoint=row['checkpoint'],checkpoint_sha256=checkpoint_sha,original_checkpoint_sha256=original_sha,utility_source_file=row.get('source_file'),inference_graph=row['graph'],
                    metrics=metrics(prob,fit['threshold']),uncertainty=paired_bootstrap(prob,a.seed),diagnostics=diag,
                    attacker=fit,attacker_sha256=sha(out/'attacker.joblib'),sampling_sha256=sampling_sha,source_hashes=hashes,
                    provenance=dict(split=identity['split'],social=identity['social'],deletion=identity['deletion']),
                    requested_users=plan['requested_users'],eligible_requested_users=plan['eligible_requested_users'],
                    fitting_users=ntrain,validation_users=nval,manifest=str(out/'manifest.json'),
                    interpretation='History-query interaction membership through deployed shared item representations; not a native deleted-user prediction or user-membership guarantee. Original-model attack strength and matched retraining are required; weak attacks are inconclusive. All methods use the same history excluding both audited items.')
        atomic_json(dest,report);print(json.dumps(dict(method=row['method'],auc=report['metrics']['auc'],users=len(prob))),flush=True)
    print('MIA_CASE',a.backbone,a.dataset,a.seed,len(rows),out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--backbone',choices=['ODSR','EGCL'],required=True);p.add_argument('--dataset',choices=['epinions','yelp2018'],required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--device',default='cpu');run(p.parse_args())
