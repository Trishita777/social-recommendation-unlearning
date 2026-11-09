"""Controlled ODSR user-deletion experiments. Run on DGL only."""
import argparse, copy, hashlib, json, math, os, sys, time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'external/ODSR'), str(ROOT/'src')]
from ODSR import ODSR
from common import seed_everything, _safe_read_pairs, _sp_to_tensor
from social_unlearning.checkpoint import save_checkpoint


def digest(x):
    return hashlib.sha256(np.asarray(x, dtype=np.int64).tobytes()).hexdigest()


def atomic_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(obj, indent=2, allow_nan=False)); os.replace(tmp, path)


def load_data(dataset, seed):
    base = ROOT/'external/ODSR/dataset'/dataset
    files = [base/'train.txt', base/'test.txt', base/('trust.txt' if (base/'trust.txt').exists() else 'links.txt')]
    for p in files:
        if not p.exists() or p.read_bytes().startswith(b'version https://git-lfs.github.com/spec/v1'):
            raise ValueError(f'Missing data or unresolved LFS: {p}')
    train_raw, test_raw, social_raw = [_safe_read_pairs(str(p)) for p in files]
    train_raw = list(dict.fromkeys(train_raw)); test_raw = list(dict.fromkeys(test_raw))
    overlap = set(train_raw)&set(test_raw)
    # Preserve supplied test; remove overlapping pairs from the training source.
    train_raw = [x for x in train_raw if x not in overlap]
    um = {u:k for k,u in enumerate(dict.fromkeys(u for u,i in train_raw))}
    im = {i:k for k,i in enumerate(dict.fromkeys(i for u,i in train_raw))}
    by = defaultdict(list)
    for u,i in train_raw: by[um[u]].append(im[i])
    rng = np.random.default_rng(seed); train=[]; val=[]
    for u, items in by.items():
        items=np.array(sorted(items)); rng.shuffle(items)
        n=min(len(items)-1, max(1, int(len(items)*.1))) if len(items)>1 else 0
        val.extend((u,int(i)) for i in items[:n]); train.extend((u,int(i)) for i in items[n:])
    test=[(um[u], im[i]) for u,i in test_raw if u in um and i in im]
    social=sorted(set((um[u],um[v]) for u,v in social_raw if u in um and v in um and u!=v))
    assert not set(train)&set(val) and not set(train)&set(test) and not set(val)&set(test)
    arr=lambda x:np.array(x,dtype=np.int64).reshape(-1,2)
    train,val,test,social=map(arr,(train,val,test,social))
    meta=dict(dataset=dataset,seed=seed,users=len(um),items=len(im),train=len(train),validation=len(val),test=len(test),raw_directed_social_edges=len(social),overlap_removed_from_train=len(overlap),test_unseen_excluded=len(test_raw)-len(test),social_only_users_excluded=len(set(x for p in social_raw for x in p)-set(um)),source_files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},split_hash=digest(np.concatenate([train, np.array([[-1,-1]]),val,np.array([[-2,-2]]),test])),social_hash=digest(social),social_policy='ODSR: symmetrize, deduplicate, add retained-user self loops, row normalize')
    return SimpleNamespace(train=train,val=val,test=test,social=social,U=len(um),I=len(im),um=um,im=im,meta=meta)


def graph(raw, keep, device):
    t0=time.perf_counter(); keep=np.asarray(keep); mapping=np.full(raw.U,-1,dtype=np.int64); mapping[keep]=np.arange(len(keep))
    mask=mapping[raw.train[:,0]]>=0; train=raw.train[mask].copy(); train[:,0]=mapping[train[:,0]]
    smask=(mapping[raw.social[:,0]]>=0)&(mapping[raw.social[:,1]]>=0)
    social=mapping[raw.social[smask]]
    U=len(keep); I=raw.I
    ui=sp.csr_matrix((np.ones(len(train),np.float32),(train[:,0],train[:,1])),shape=(U,I))
    adj=sp.bmat([[None,ui],[ui.T,None]],format='csr'); deg=np.asarray(adj.sum(1)).ravel(); inv=np.zeros_like(deg); np.power(deg,-.5,out=inv,where=deg>0)
    norm=sp.diags(inv)@adj@sp.diags(inv)
    soc=sp.csr_matrix((np.ones(len(social),np.float32),(social[:,0],social[:,1])),shape=(U,U)); soc=soc.maximum(soc.T); soc.setdiag(1); soc.eliminate_zeros()
    ds=np.asarray(soc.sum(1)).ravel(); uu=sp.diags(1/ds)@soc
    positive=[set() for _ in keep]
    for u,i in train: positive[u].add(int(i))
    def mapped(pairs):
        p=pairs[mapping[pairs[:,0]]>=0].copy(); p[:,0]=mapping[p[:,0]]; return p
    out=SimpleNamespace(user_num=U,item_num=I,adj_ui=_sp_to_tensor(ui,device),adj_uu=_sp_to_tensor(uu,device),norm_adj=_sp_to_tensor(norm,device),train=train,val=mapped(raw.val),test=mapped(raw.test),keep=keep,mapping=mapping,positives=positive,ui=ui,social=social)
    out.counts=dict(users=U,interactions=len(train),directed_social_edges=len(social),removed_users=raw.U-U,removed_interactions=len(raw.train)-len(train),removed_directed_social_edges=len(raw.social)-len(social),isolated_items=int((np.asarray(ui.sum(0)).ravel()==0).sum()),isolated_users=int((np.asarray(ui.sum(1)).ravel()==0).sum()),graph_seconds=time.perf_counter()-t0)
    return out


def negatives(data, users, rng):
    n=rng.integers(data.item_num,size=len(users))
    for k,u in enumerate(users):
        if len(data.positives[u])==data.item_num: raise ValueError('No negative item')
        while int(n[k]) in data.positives[u]: n[k]=rng.integers(data.item_num)
    return n


@torch.no_grad()
def evaluate(model,data,pairs,device,reference=None,groups=None):
    model.eval(); U,I=model(); truth=defaultdict(set)
    for u,i in pairs: truth[int(u)].add(int(i))
    totals=defaultdict(float); per={}; ranks={}; n=0
    for start in range(0,len(truth),128):
        users=list(truth)[start:start+128]; scores=U[users]@I.T
        for row,u in enumerate(users):
            if data.positives[u]: scores[row,list(data.positives[u])]=-torch.inf
        ids=scores.topk(min(20,data.item_num),dim=1).indices.cpu().numpy()
        for u,top in zip(users,ids):
            n+=1; hit=np.array([i in truth[u] for i in top]); row={}
            for k in (10,20):
                h=hit[:k]; denom=min(len(truth[u]),k)
                row[f'precision@{k}']=float(h.sum()/k); row[f'recall@{k}']=float(h.sum()/len(truth[u]))
                row[f'map@{k}']=float(((np.cumsum(h)/np.arange(1,len(h)+1))*h).sum()/denom)
                row[f'ndcg@{k}']=float((h/np.log2(np.arange(2,len(h)+2))).sum()/(1/np.log2(np.arange(2,denom+2))).sum())
            for key,value in row.items(): totals[key]+=value
            per[u]=row; ranks[u]=top
    result={key:value/max(n,1) for key,value in totals.items()}; result['evaluated_users']=n
    if groups is not None:
        result['groups']={name:dict(users=len(us),**{key:float(np.mean([per[u][key] for u in us])) for key in totals}) for name, us in {name:[u for u in per if groups[u]==name] for name in set(groups.values())}.items() if us}
    if reference is not None:
        ru,ri,rr=reference; errors=[]; agreements=[]; grouped=defaultdict(lambda: [[],[]])
        # fixed item probe; same catalog and users for all compared models
        probe=torch.arange(0,data.item_num,max(1,data.item_num//1024),device=device)
        for start in range(0,len(truth),128):
            users=list(truth)[start:start+128]; a=(U[users]@I[probe].T); b=ru[users]@ri[probe].T
            rmse=((a-b).square().mean(1).sqrt()).cpu().tolist()
            for u,e in zip(users,rmse):
                agr=len(set(ranks[u])&set(rr[u]))/len(ranks[u]); errors.append(e); agreements.append(agr)
                if groups is not None: grouped[groups[u]][0].append(e); grouped[groups[u]][1].append(agr)
        result['fidelity']=dict(score_rmse=float(np.mean(errors)),top20_overlap=float(np.mean(agreements)),score_probe_items=len(probe),groups={g:dict(users=len(x[0]),score_rmse=float(np.mean(x[0])),top20_overlap=float(np.mean(x[1]))) for g,x in grouped.items()})
    return result,(U.detach(),I.detach(),ranks)


def compact(original,data,cfg,device):
    model=ODSR(data,cfg,device).to(device); state={k:v.detach().clone() for k,v in original.state_dict().items()}; state['embedding_dict.user_emb']=state['embedding_dict.user_emb'][data.keep]; model.load_state_dict(state); return model


@torch.no_grad()
def sensitivity(original,teacher,data,rng,device,batch=512):
    t0=time.perf_counter(); original.eval(); teacher.eval(); ou,oi=original(); tu,ti=teacher(); oldkeep=torch.as_tensor(data.keep,device=device)
    # Fixed negatives: same triplets on the original and counterfactual retained graphs.
    neg=negatives(data,data.train[:,0],rng); rec=np.zeros(len(data.train),np.float32)
    for s in range(0,len(rec),4096):
        u,p=torch.as_tensor(data.train[s:s+4096],device=device).T; n=torch.as_tensor(neg[s:s+4096],device=device)
        before=(ou[oldkeep[u]]*(oi[p]-oi[n])).sum(1); after=(tu[u]*(ti[p]-ti[n])).sum(1)
        rec[s:s+4096]=(before-after).abs().cpu().numpy()
    retained=rng.choice(data.user_num,min(512,data.user_num),replace=False); deleted=np.setdiff1d(np.arange(original.user_num),data.keep)
    deleted=rng.choice(deleted,min(512,len(deleted)),replace=False)
    # Common retained candidate distributions plus exact deleted mass within this declared candidate probe.
    oc=F.normalize(ou[oldkeep[torch.as_tensor(retained,device=device)]],dim=1); tc=F.normalize(tu[retained],dim=1); dc=F.normalize(ou[deleted],dim=1)
    anchor=np.zeros(data.user_num,np.float32); targets=[]
    for s in range(0,data.user_num,batch):
        a=F.normalize(ou[oldkeep[s:s+batch]],dim=1); b=F.normalize(tu[s:s+batch],dim=1)
        lo=a@oc.T/teacher.ssl_temp; lt=b@tc.T/teacher.ssl_temp
        po=lo.softmax(1); pt=lt.softmax(1)
        js=.5*((po*(po.clamp_min(1e-12).log()-((po+pt)/2).clamp_min(1e-12).log())).sum(1)+(pt*(pt.clamp_min(1e-12).log()-((po+pt)/2).clamp_min(1e-12).log())).sum(1))
        mass=torch.cat([lo,a@dc.T/teacher.ssl_temp],1).softmax(1)[:,lo.shape[1]:].sum(1) if len(deleted) else torch.zeros(len(a),device=device)
        anchor[s:s+batch]=(js+mass).cpu().numpy(); targets.append(pt)
    def weight(x):
        # Normalize sensitivity first, clip to [0,4], then add a positive floor and normalize mean.
        w=1+np.clip(x/max(float(x.mean()),1e-8),0,4); return w/w.mean()
    return dict(rec=torch.tensor(weight(rec),device=device),anchor=torch.tensor(weight(anchor),device=device),targets=torch.cat(targets),candidates=torch.as_tensor(retained,device=device),seconds=time.perf_counter()-t0,stats=dict(rec_mean=float(rec.mean()),anchor_mean=float(anchor.mean()),retained_candidates=len(retained),deleted_candidates=len(deleted),deleted_mass_scope='fixed sampled candidate probe, not full user universe'))


def loss_terms(model,u,p,n,weights=None,indices=None,mode='original'):
    fu,fi=model(); rec=F.softplus(-((fu[u]*(fi[p]-fi[n])).sum(1)))
    av,_=model('adaptive'); bv,_=model(); v1=F.normalize(av[u],dim=1); v2=F.normalize(bv[u],dim=1)
    # Preserve official ODSR's batch candidate pool, including repeated-user multiplicity.
    logits=v1@v2.T/model.ssl_temp; cl=F.cross_entropy(logits,torch.arange(len(u),device=u.device),reduction='none')
    rw=cw=torch.ones_like(rec)
    if weights is not None and mode not in ('uniform','retain_finetune'):
        rw=weights['rec'][indices]
        if mode!='rec_only': cw=weights['anchor'][u]
    loss=(rec*rw).mean()+model.ssl_reg*(cl*cw).mean()+model.reg*(fu[u].norm()+fi[p].norm()+fi[n].norm())
    if weights is not None and mode not in ('weighted_finetune','no_correction','retain_finetune'):
        # Counterfactual retained-graph teacher targets may still contain learned influence.
        logits=F.normalize(fu[u],dim=1)@F.normalize(fu[weights['candidates']],dim=1).T/model.ssl_temp
        kl=F.kl_div(logits.log_softmax(1),weights['targets'][u],reduction='none').sum(1)
        loss=loss+model.ssl_reg*(kl*cw).mean()
    return loss


def train(model,data,cfg,out,provenance,epochs,mode,device,weights=None):
    out.mkdir(parents=True,exist_ok=True); rng=np.random.default_rng(cfg['seed']); opt=torch.optim.Adam(model.parameters(),lr=cfg['lr']); best=-1.; bad=0; steps=0; t0=time.perf_counter(); history=[]; best_state=None; optimizer_seconds=0.
    for ep in range(1,epochs+1):
        model.train(); order=rng.permutation(len(data.train)); total=0.; t=time.perf_counter()
        for start in range(0,len(order),cfg['batch_size']):
            ids=order[start:start+cfg['batch_size']]; b=data.train[ids]; n=negatives(data,b[:,0],rng)
            u,p=torch.as_tensor(b,device=device).T; n=torch.as_tensor(n,device=device)
            opt.zero_grad(set_to_none=True); loss=loss_terms(model,u,p,n,weights,torch.as_tensor(ids,device=device),mode)
            if not torch.isfinite(loss): raise ValueError(f'Nonfinite {mode} loss')
            loss.backward(); opt.step(); steps+=1; total+=float(loss.detach())*len(b)
        torch.cuda.synchronize(device); optimizer_seconds+=time.perf_counter()-t
        metrics,_=evaluate(model,data,data.val,device); score=metrics['recall@20']; improved=score>best
        if improved: best=score; bad=0; best_state=copy.deepcopy(model.state_dict()); best_ep=ep
        else: bad+=1
        row=dict(epoch=ep,loss=total/len(order),validation=metrics,elapsed_seconds=time.perf_counter()-t0); history.append(row)
        save_checkpoint(out/'last.pt',model,opt,config=cfg,provenance=provenance,epoch=ep,step=steps,metrics=metrics,complete=False)
        if improved: save_checkpoint(out/'best_val.pt',model,opt,config=cfg,provenance=provenance,epoch=ep,step=steps,metrics=metrics,complete=False)
        if ep%5==0: save_checkpoint(out/f'epoch_{ep:03}.pt',model,opt,config=cfg,provenance=provenance,epoch=ep,step=steps,metrics=metrics,complete=False)
        atomic_json(out/'history.json',history); print(json.dumps(dict(method=mode,**row)),flush=True)
        if bad>=cfg['patience']: break
    save_checkpoint(out/'last.pt',model,opt,config=cfg,provenance=provenance,epoch=ep,step=steps,metrics=metrics,complete=True)
    model.load_state_dict(best_state)
    # Final deployment artifact has compact rows, graph is reconstructed from retained data, no optimizer/teacher.
    temp=out/'deployment.tmp'; torch.save(dict(model=model.state_dict(),retained_original_ids=data.keep,config=cfg,provenance=provenance,best_epoch=best_ep),temp); os.replace(temp,out/'deployment.pt')
    return dict(training_seconds=time.perf_counter()-t0,optimizer_seconds=optimizer_seconds,best_epoch=best_ep,epochs=ep,steps=steps,convergence='validation_patience' if bad>=cfg['patience'] else 'configured_epoch_budget',epoch_budget=epochs)


def run(args):
    torch.set_num_threads(8); device=torch.device(args.device); seed_everything(args.seed); raw=load_data(args.dataset,args.seed)
    cfg=dict(seed=args.seed,lr=.001,emb_size=64,n_layer=3,tau=.2,reg=0.,eps=.46 if args.dataset=='epinions' else .54,ssl_reg=.18 if args.dataset=='epinions' else .43,dropout=.14 if args.dataset=='epinions' else .17,batch_size=args.batch_size,patience=10,correction_coefficient='ssl_reg',weight_floor=1,weight_clip=4,weight_probe_candidates=512,original_epochs=args.original_epochs,unlearn_epochs=args.unlearn_epochs)
    source_files=[Path(__file__),ROOT/'external/ODSR/ODSR.py',ROOT/'src/social_unlearning/checkpoint.py']
    source=hashlib.sha256(b''.join(p.read_bytes() for p in source_files)).hexdigest(); provenance=dict(split=raw.meta['split_hash'],data=hashlib.sha256(json.dumps(raw.meta['source_files'],sort_keys=True).encode()).hexdigest(),source=source,social=raw.meta['social_hash'],deletion='original')
    base=ROOT/'checkpoints/ODSR'/args.dataset/f'seed_{args.seed}'/raw.meta['split_hash'][:16]; origdir=base/'original'; orig=graph(raw,np.arange(raw.U),device)
    atomic_json(base/'data_manifest.json',dict(**raw.meta,user_map=raw.um,item_map=raw.im,config=cfg,source_hash=source))
    seed_everything(args.seed); original=ODSR(orig,cfg,device).to(device)
    if (origdir/'deployment.pt').exists():
        ck=torch.load(origdir/'deployment.pt',map_location=device,weights_only=False)
        if ck['provenance']!=provenance or ck['config']!=cfg: raise ValueError('Original cache provenance mismatch')
        original.load_state_dict(ck['model']); original_time=json.loads((origdir/'timing.json').read_text())
    else:
        original_time=train(original,orig,cfg,origdir,provenance,args.original_epochs,'original',device); atomic_json(origdir/'timing.json',original_time)
    deletion_order=np.random.default_rng(args.seed+10000).permutation(raw.U)
    if args.strategy=='degree':
        degree=np.bincount(raw.social.ravel(),minlength=raw.U); deletion_order=np.lexsort((np.arange(raw.U),-degree))
    for fraction in args.fractions:
        t0=time.perf_counter(); torch.cuda.reset_peak_memory_stats(device); count=max(1,int(round(raw.U*fraction))); deleted=np.sort(deletion_order[:count]); keep=np.setdiff1d(np.arange(raw.U),deleted); data=graph(raw,keep,device)
        deletion=digest(deleted); prov=dict(provenance,deletion=deletion); out=base/deletion[:16]/args.strategy; out.mkdir(parents=True,exist_ok=True)
        atomic_json(out/'deletion.json',dict(fraction=fraction,strategy=args.strategy,deleted_users=deleted.tolist(),counts=data.counts,provenance=prov))
        affected=set(raw.social[np.isin(raw.social,deleted).any(1)].ravel())-set(deleted)
        groups={u:'social_neighbor' if int(old) in affected else 'control' for u,old in enumerate(keep)}
        seed_everything(args.seed); retrain=ODSR(data,cfg,device).to(device)
        rt=train(retrain,data,cfg,out/'retrain',prov,args.original_epochs,'retrain',device); metrics,ref=evaluate(retrain,data,data.test,device,groups=groups)
        results=dict(dataset=args.dataset,backbone='ODSR',seed=args.seed,fraction=fraction,strategy=args.strategy,deletion_hash=deletion,split_hash=raw.meta['split_hash'],counts=data.counts,proposed_status='experimental objective; counterfactual teacher is not scratch retraining',mia_status='pending_validated_user_level_shadow_attack',coverage_scope='ODSR controls and proposed ablations; published baselines and EGCL pending',methods={'retrain':dict(metrics=metrics,**rt,total_deletion_seconds=data.counts['graph_seconds']+rt['training_seconds'])})
        # Original predictions evaluated on identical retained data/user catalog.
        teacher_start=time.perf_counter(); teacher=compact(original,data,cfg,device); teacher_setup_seconds=time.perf_counter()-teacher_start; original.eval()
        class Restricted:
            def eval(self): return self
            def __call__(self):
                u,i=original(); return u[data.keep],i
        om,_=evaluate(Restricted(),data,data.test,device,reference=ref,groups=groups); results['methods']['original']=dict(metrics=om,**original_time)
        rng=np.random.default_rng(args.seed); weights=sensitivity(original,teacher,data,rng,device); weights['seconds']+=teacher_setup_seconds; atomic_json(out/'weight_summary.json',dict(seconds=weights['seconds'],**weights['stats']))
        for method in args.methods:
            seed_everything(args.seed); model=compact(original,data,cfg,device)
            tm=train(model,data,cfg,out/method,prov,args.unlearn_epochs,method,device,weights=None if method=='retain_finetune' else weights)
            metrics,_=evaluate(model,data,data.test,device,reference=ref,groups=groups)
            pre=data.counts['graph_seconds']+(0 if method=='retain_finetune' else weights['seconds'])
            results['methods'][method]=dict(metrics=metrics,**tm,preprocessing_weight_seconds=pre,total_deletion_seconds=pre+tm['training_seconds'],speedup_vs_retrain=(rt['training_seconds']+data.counts['graph_seconds'])/(pre+tm['training_seconds']))
            atomic_json(out/'results.json',results); del model
        results['peak_gpu_memory_bytes']=torch.cuda.max_memory_allocated(device); results['experiment_wall_seconds']=time.perf_counter()-t0; results['status']='completed_declared_odsr_subset'
        atomic_json(out/'results.json',results); print('RESULT '+str(out/'results.json'),flush=True)
        del retrain,teacher,weights,ref


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--dataset',default='epinions'); p.add_argument('--seed',type=int,default=42); p.add_argument('--device',default='cuda:1'); p.add_argument('--fractions',nargs='+',type=float,default=[.01]); p.add_argument('--strategy',choices=['random','degree'],default='random'); p.add_argument('--original-epochs',type=int,default=30); p.add_argument('--unlearn-epochs',type=int,default=5); p.add_argument('--batch-size',type=int,default=2048); p.add_argument('--methods',nargs='+',default=['retain_finetune','weighted_finetune','proposed','uniform','rec_only']); run(p.parse_args())
