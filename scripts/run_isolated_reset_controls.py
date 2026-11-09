"""Postprocess completed controls in a new namespace; reuse retained training.

Reset-after-finetuning avoids redoing native training. Its cost includes the
source checkpoint's recorded deletion cost, plus separately timed reset work.
Existing result tables and checkpoints are never overwritten.
"""
import argparse
import fcntl
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from run_unlearning import ROOT,ODSR,load_data,digest,evaluate,atomic_json
from social_unlearning.edge_deletion import edge_graph
from social_unlearning.isolated_reset import reset_isolated_odsr_rows
from social_unlearning.benchmark_mio import BenchmarkMIO,embedding_features
from audit_corrected_forgetting import probes,score_probe,compare


def run(a):
    torch.set_num_threads(4);device=torch.device(a.device);torch.cuda.set_device(device)
    raw=load_data(a.dataset,a.seed)
    deleted=np.sort(np.random.default_rng(a.seed+10000).permutation(raw.U)[:max(1,round(raw.U*.01))])
    relative=Path('ODSR')/a.dataset/f'seed_{a.seed}'/raw.meta['split_hash'][:16]/digest(deleted)[:16]
    source_dir=ROOT/'checkpoints/edge_user_v1'/relative/a.source_method
    out=ROOT/'checkpoints/edge_user_reset_v1'/relative/f'reset_{a.source_method}'
    out.mkdir(parents=True,exist_ok=True)
    with (out/'run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        source=hashlib.sha256(Path(__file__).read_bytes()+(ROOT/'src/social_unlearning/isolated_reset.py').read_bytes()).hexdigest()
        if (out/'results.json').exists():
            old=json.loads((out/'results.json').read_text())
            if old['implementation_source']!=source:raise ValueError('Reset source changed')
            print('REUSE',out,flush=True);return
        ck=torch.load(source_dir/'deployment.pt',map_location='cpu',weights_only=False)
        source_report=json.loads((source_dir/'results.json').read_text())
        assert ck['provenance']['deletion_semantics']=='incident_edges_only_keep_user_rows'
        data=edge_graph(raw,deleted,device);model=ODSR(data,ck['config'],device).to(device)
        model.load_state_dict(ck['model']);model.eval()
        torch.cuda.synchronize(device);start=time.perf_counter()
        reset=reset_isolated_odsr_rows(model,data,deleted,1000003+a.seed)
        torch.cuda.synchronize(device);reset_seconds=time.perf_counter()-start
        # Structural check, independent of nondeterministic CUDA sparse-reduction
        # order. Re-evaluated ranks can differ at near-ties from an older report.
        active=np.setdiff1d(np.arange(raw.U),deleted)
        for name,value in model.state_dict().items():
            actual=value.detach().cpu();expected=ck['model'][name].cpu()
            if name=='embedding_dict.user_emb':actual=actual[active];expected=expected[active]
            if not torch.equal(actual,expected):
                raise AssertionError(f'Reset changed nonrequested parameters: {name}')
        prov=dict(ck['provenance'],source=source,source_method=a.source_method,
            source_checkpoint=str(source_dir/'deployment.pt'),reset=reset)
        torch.save(dict(model=model.state_dict(),retained_original_ids=np.arange(raw.U),deleted_user_ids=deleted,
            config=ck['config'],provenance=prov),out/'deployment.tmp')
        (out/'deployment.tmp').replace(out/'deployment.pt')
        ref_dir=ROOT/'checkpoints/edge_user_v1'/relative/'retrain'
        reference_ck=torch.load(ref_dir/'deployment.pt',map_location='cpu',weights_only=False)
        reference_model=ODSR(data,reference_ck['config'],device).to(device)
        reference_model.load_state_dict(reference_ck['model'])
        _,reference=evaluate(reference_model,data,data.test,device)
        metrics,emb=evaluate(model,data,data.test,device,reference=reference)
        probe=probes(raw,deleted,a.seed)
        records=score_probe((emb[0].cpu(),emb[1].cpu()),probe)
        ref_records=score_probe((reference[0].cpu(),reference[1].cpu()),probe)
        forgetting=dict(compare(records,ref_records),positive_beats_negative=float(np.mean([r['positive_beats_negative'] for r in records])))
        delta={k:metrics[k]-source_report['metrics'][k] for k in ['recall@20','ndcg@20','precision@20','map@20']}
        mia=None
        mia_path=ROOT/'results/benchmark_mio/ODSR'/a.dataset/f'seed_{a.seed}'/'results.json'
        if mia_path.exists():
            existing=json.loads(mia_path.read_text());saved=torch.load(existing['attacker'],map_location='cpu',weights_only=False)
            attacker=BenchmarkMIO(emb[0].shape[1]*2);attacker.load_state_dict(saved['model'])
            histories=defaultdict(list)
            for u,i in np.concatenate([raw.train,raw.val]):histories[int(u)].append(int(i))
            x=embedding_features(emb[0].cpu(),emb[1].cpu(),deleted,{u:u for u in range(raw.U)},histories)
            probability=attacker.member_probability(x).numpy()
            mia=dict(mean_membership_probability=float(probability.mean()),member_classification_rate=float((probability>=.5).mean()),
                scope='positive-only diagnostic, not target AUC/accuracy',attacker=existing['attacker'])
        report=dict(status='completed_isolated_reset_control',implementation_source=source,backbone='ODSR',dataset=a.dataset,seed=a.seed,fraction=.01,
            method=f'reset_{a.source_method}',provenance=prov,metrics=metrics,retained_utility_delta=delta,
            invariance_check='all nonrequested parameters bitwise unchanged; isolated-graph invariance tested on CPU',
            numerical_note='Delta is against a historical GPU evaluation; CUDA sparse reductions may change near-tie ranks',
            forgetting=forgetting,mio=mia,efficiency=dict(reset_seconds=reset_seconds,
                source_method_wall_seconds=source_report['efficiency']['wall_seconds'],
                composed_seconds=source_report['efficiency']['wall_seconds']+reset_seconds,
                timing_mode='reused source timing plus reset only; not a new isolated end-to-end measurement'),
            limitation='Independent requested-row reset is a control; no guarantee of removing shared item/gate/pooling influence')
        atomic_json(out/'results.json',report);print('RESET_RESULT',json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--source-method',choices=['graph_only','retain_finetune','ours'],required=True);p.add_argument('--device',default='cuda:0')
    run(p.parse_args())
