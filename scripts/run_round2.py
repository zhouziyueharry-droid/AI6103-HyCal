"""Round2: development-only search, frozen choices, then confirmation and test.

All outputs are separate from the immutable first run. DTD has no tuning phase.
"""
import argparse,csv,hashlib,json,os,random,statistics,sys,time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import torch
import hycal_core as core
import round2_core as r2
from run_experiments import prepare_stats,invert

OUT=ROOT/'results/part_a_round2'
CONFIG=ROOT/'configs/round2.json'
PARTITION=ROOT/'data/manifests/flowers102/round2_validation.json'


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat()
def save(p,data):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8')
def identity(cfg): return cfg['family']+'_'+hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()[:10]


def candidates():
    configs=[{'family':f} for f in ['H0','H2','MD','CLIP','VisualPrototype']]
    weight0=1/(1+__import__('math').exp(1))
    configs += [{'family':'H2_weight','weight':w} for w in [.5,1.]]
    configs += [{'family':'GDA','alpha':a} for a in [.0001,.001,.01,.1,1.,10.,100.]]
    configs += [{'family':'TipAdapter','alpha':a,'beta':b} for a in [.1,1.,10.,100.] for b in [1.,5.,10.,20.]]
    configs += [{'family':'ProKeR','lam':l,'beta':b} for l in [.001,.01,.1,1.,10.] for b in [.1,1.,5.,10.]]
    configs += [{'family':'RDA_global' if k==0 else 'RDA_semantic','neighbors':k,'alpha':a,'ridge':l,'weight':w}
                for k in [0,8,16] for a in [.5,1.] for l in [1e-5,1e-4,1e-3] for w in [weight0,.5,1.]]
    configs += [{'family':'RDA_transport','transport':True,'neighbors':k,'alpha':a,'ridge':l,'weight':w}
                for k in [0,8,16] for a in [.5,1.] for l in [1e-4,1e-3] for w in [.5,1.]]
    return configs


def initialize():
    if CONFIG.exists() or PARTITION.exists(): raise RuntimeError('Round2 protocol already initialized')
    records=json.loads((ROOT/'data/manifests/flowers102/records.json').read_text())
    groups={}
    for r in records:
        if r['split']=='val': groups.setdefault(r['label'],[]).append(r['sample_id'])
    parts={'dev':[],'confirmation':[]}
    rng=random.Random(20261116)
    for label in sorted(groups):
        ids=sorted(groups[label]);rng.shuffle(ids)
        parts['dev']+=ids[:len(ids)//2];parts['confirmation']+=ids[len(ids)//2:]
    assert not set(parts['dev'])&set(parts['confirmation'])
    save(PARTITION,{'created_utc':now(),'seed':20261116,'indices':parts,
                    'scope':'Both halves were previously used as round1 validation. This only separates round2 adaptive use.'})
    save(CONFIG,{'created_utc':now(),'shot':5,'selection_seeds':[0,1,42,1993],
                 'final_seeds':[0,1,42,1993]+list(range(2,18)),'candidates':candidates(),
                 'selection':'highest four-seed development macro per family; deterministic candidate order for ties',
                 'confirmation_policy':'one evaluation after selection freeze; no reselection',
                 'test_policy':'Flowers exploratory because round1 test known; DTD transfer with unchanged parameters',
                 'partition_sha256':sha(PARTITION),'first_round_lock_sha256':sha(ROOT/'results/part_a/execution_lock.json')})
    print(json.dumps({'candidates':len(candidates()),'dev':len(parts['dev']),'confirmation':len(parts['confirmation'])}))


def inputs(dataset,phase):
    folder=ROOT/'data/manifests'/dataset
    records=json.loads((folder/'records.json').read_text())
    cache=torch.load(ROOT/f'artifacts/features/{dataset}/features.pt',map_location='cpu',weights_only=False)
    assert cache['sample_ids']==[r['sample_id'] for r in records]
    assert cache['metadata']['manifest_sha256']==sha(folder/'records.json')
    support=json.loads((folder/'support_indices.json').read_text())
    if dataset=='flowers102':
        rep=json.loads((folder/'replication_support_indices.json').read_text())
        support=dict(support,indices={**support['indices'],**rep['indices']})
    if phase in ['dev','confirmation']:
        if dataset!='flowers102': raise ValueError('DTD cannot tune or use confirmation')
        ids=set(json.loads(PARTITION.read_text())['indices'][phase])
        query_idx=[i for i,r in enumerate(records) if r['sample_id'] in ids]
    else: query_idx=[i for i,r in enumerate(records) if r['split']=='test']
    return cache,records,support,query_idx


def context(cache,records,support,query_idx,seed,device):
    lookup={r['sample_id']:i for i,r in enumerate(records)}
    ids=support['indices'][str(seed)]['5']
    bylabel={c['label']:[] for c in cache['classes']}
    for sid in ids:
        i=lookup[sid];assert records[i]['split']=='train'
        bylabel[records[i]['label']].append(i)
    index=torch.tensor([bylabel[c['label']] for c in cache['classes']])
    assert index.shape==(len(cache['classes']),5)
    raw=cache['image_features'][index].to(device)
    proto,cov,texts,(bg_mu,bg_cov)=prepare_stats(cache,records,support,seed,5,device)
    bg_p=torch.pinverse(bg_cov+1e-4*torch.eye(bg_cov.shape[0],device=device))
    return {'raw':raw,'flat':raw.flatten(0,1),'support_labels':torch.arange(len(texts),device=device).repeat_interleave(5),
            'texts':texts,'proto':proto,'cov':cov,'background':(bg_mu,bg_p),
            'query':cache['image_features'][query_idx].to(device),'true':cache['labels'][query_idx],
            'candidate_labels':torch.tensor([c['label'] for c in cache['classes']]),'memo':{}}


@torch.no_grad()
def scores(ctx,cfg):
    family=cfg['family'];q=ctx['query'];x=ctx['flat'];t=ctx['texts'];memo=ctx['memo']
    if family=='CLIP': return (q@t.T).cpu()
    if family=='VisualPrototype': return (q@core.normalize_embeddings(ctx['raw'].mean(1)).T).cpu()
    if family=='TipAdapter': return r2.tip_scores(q,x,ctx['support_labels'],t,cfg['beta'],cfg['alpha']).cpu()
    if family=='ProKeR':
        b=r2.proker_fit(x,ctx['support_labels'],t,cfg['beta'],cfg['lam'])
        return r2.proker_scores(q,x,t,b,cfg['beta']).cpu()
    if family=='GDA':
        if 'gda' not in memo:
            mean,p=r2.gda_parameters(ctx['raw']);memo['gda']=r2.gda_scores(q,mean,p).cpu()
        return 100*(q@t.T).cpu()+cfg['alpha']*memo['gda']
    if family in ['H0','H2','H2_weight','MD']:
        rho=0. if family in ['H0','MD'] else .5
        key=('baseline',rho)
        if key not in memo:
            p=invert(core.regularize_covariance(ctx['cov'],rho,1e-4))
            memo[key]=core.score_features(q,t,ctx['proto'],p,128,ctx['background'])
        cos,md=memo[key]
        if family=='H0': return cos
        if family=='MD': return md
        w=cfg.get('weight',1/(1+__import__('math').exp(1)))
        return (1-w)*cos+w*md
    if family.startswith('RDA'):
        key=tuple((k,json.dumps(v,sort_keys=True)) for k,v in sorted(cfg.items()) if k not in ['weight','family'])
        if key not in memo:
            covariance=ctx['cov']
            if cfg.get('rawmean_center',False):
                fused=core.normalize_embeddings(ctx['raw']+t[:,None,:])
                z=fused-fused.mean(1,keepdim=True)
                covariance=torch.stack([a.T@a/4 for a in z])
            if cfg.get('transport',False):
                fused=core.normalize_embeddings(ctx['raw']+t[:,None,:])
                residuals=fused-ctx['proto'][:,None,:]
                prior=r2.transported_prior(residuals,ctx['proto'],t,cfg['neighbors'])
                p_cov=(1-cfg['alpha'])*covariance+cfg['alpha']*prior+cfg['ridge']*torch.eye(covariance.shape[-1],device=q.device)
            else:
                p_cov=r2.rda_covariance(covariance,t,cfg['alpha'],cfg['neighbors'],cfg['ridge'],cfg.get('neighbor_mode','semantic'))
            p=invert(p_cov)
            logdet=torch.linalg.slogdet(p_cov)[1] if cfg.get('logdet',False) else None
            cos,md=r2.quadratic_scores(q,t,ctx['proto'],p,128,logdet)
            memo[key]=(core.normalize_scores(cos).cpu(),core.normalize_scores(md).cpu())
        cos,md=memo[key];w=cfg['weight']
        return (1-w)*cos+w*md
    raise ValueError(family)


def provenance():
    files=['src/round2_core.py','scripts/run_round2.py','configs/round2.json',
           'data/manifests/flowers102/round2_validation.json','results/part_a/execution_lock.json',
           'src/hycal_core.py','scripts/run_experiments.py',
           'data/manifests/flowers102/records.json','data/manifests/flowers102/support_indices.json',
           'data/manifests/flowers102/replication_support_indices.json',
           'artifacts/features/flowers102/features.pt']
    return {p:sha(ROOT/p) for p in files}


def run(dataset,phase,config_list,seeds,save_predictions=False):
    cache,records,support,query_idx=inputs(dataset,phase)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    folder=OUT/dataset/phase;folder.mkdir(parents=True,exist_ok=True)
    dataset_files=[ROOT/f'data/manifests/{dataset}/{f}' for f in ['records.json','classes.json','support_indices.json']]
    dataset_files.append(ROOT/f'artifacts/features/{dataset}/features.pt')
    binding={str(p.relative_to(ROOT)):sha(p) for p in dataset_files}
    binding.update({'source:'+k:v for k,v in provenance().items()})
    run_lock=folder/'run_lock.json'
    if run_lock.exists():
        if json.loads(run_lock.read_text())['files']!=binding: raise RuntimeError('Run inputs changed')
    else:
        save(run_lock,{'created_utc':now(),'files':binding,'dataset':dataset,'phase':phase})
    rows=[]
    for seed in seeds:
        started=time.perf_counter();ctx=context(cache,records,support,query_idx,seed,device)
        for cfg in config_list:
            tag=f's{seed}_{identity(cfg)}';target=folder/(tag+'.json')
            if target.exists():
                existing=json.loads(target.read_text())
                assert existing['config']==cfg
                rows.append(existing);continue
            tick=time.perf_counter()
            sc=scores(ctx,cfg)
            assert torch.isfinite(sc).all()
            pred=ctx['candidate_labels'][sc.argmax(1)]
            metrics=core.classification_metrics(pred,ctx['true'],len(ctx['texts']))
            row={'tag':tag,'dataset':dataset,'phase':phase,'seed':seed,'shot':5,'config':cfg,
                 'macro':metrics['macro'],'top1':metrics['top1'],'n':len(query_idx),'computed_utc':now(),
                 'seconds_including_cache_reuse':time.perf_counter()-tick}
            if save_predictions:
                row['per_class']=metrics['per_class'];row['class_counts']=metrics['class_counts']
                with (folder/(tag+'_predictions.csv')).open('w',newline='',encoding='utf-8') as f:
                    writer=csv.writer(f);writer.writerow(['sample_id','true_label','pred_label','correct'])
                    writer.writerows((records[i]['sample_id'],int(y),int(p),int(y==p)) for i,y,p in zip(query_idx,ctx['true'],pred))
            save(target,row);rows.append(row)
        print(json.dumps({'dataset':dataset,'phase':phase,'seed':seed,'configs':len(config_list),'seconds':time.perf_counter()-started}),flush=True)
        del ctx
    save(folder/'summary.json',rows)
    return rows


def freeze(rows,cfg):
    selected=[];ranking=[]
    for candidate in cfg['candidates']:
        group=[r for r in rows if r['config']==candidate]
        assert sorted(r['seed'] for r in group)==sorted(cfg['selection_seeds'])
        ranking.append({'config':candidate,'macro':statistics.mean(r['macro'] for r in group),
                        'top1':statistics.mean(r['top1'] for r in group)})
    for family in dict.fromkeys(c['family'] for c in cfg['candidates']):
        selected.append(max((r for r in ranking if r['config']['family']==family),key=lambda r:r['macro']))
    # Fixed targeted ablations tied to the selected covariance recipe, without further selection.
    sem=next(r['config'] for r in selected if r['config']['family']=='RDA_semantic')
    selected += [{'config':dict(sem,family='RDA_random_ablation',neighbor_mode='random'),'selected_by':'matched semantic config'},
                 {'config':dict(sem,family='RDA_local_ablation',alpha=0.),'selected_by':'matched semantic config'},
                 {'config':dict(sem,family='RDA_global_matched',neighbors=0),'selected_by':'matched semantic config'},
                 {'config':dict(sem,family='RDA_rawmean_ablation',rawmean_center=True),'selected_by':'matched semantic config'},
                 {'config':dict(sem,family='RDA_logdet_ablation',logdet=True),'selected_by':'matched semantic config'}]
    transported=next(r['config'] for r in selected if r['config']['family']=='RDA_transport')
    selected += [{'config':dict(transported,family='RDA_untransported_matched',transport=False),'selected_by':'matched transported config'}]
    selection={'frozen_utc':now(),'provenance':provenance(),'selected':selected,'ranking':sorted(ranking,key=lambda r:r['macro'],reverse=True)}
    save(OUT/'selection.json',selection)
    print(json.dumps({'selected':selected},indent=2),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['init','dev','confirmation','test'],required=True)
    p.add_argument('--dataset',choices=['flowers102','dtd'],default='flowers102');args=p.parse_args()
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.phase=='init': initialize();return
    cfg=json.loads(CONFIG.read_text())
    if args.phase=='dev':
        if (OUT/'selection.json').exists(): raise RuntimeError('Already frozen')
        if args.dataset!='flowers102': raise ValueError('Only Flowers development permitted')
        launch=OUT/'development_lock.json'
        if launch.exists():
            if json.loads(launch.read_text())['provenance']!=provenance():
                raise RuntimeError('Development inputs changed; archive the entire old run before restarting')
        else:
            save(launch,{'created_utc':now(),'provenance':provenance()})
        rows=run(args.dataset,'dev',cfg['candidates'],cfg['selection_seeds'])
        freeze(rows,cfg)
    else:
        selection=json.loads((OUT/'selection.json').read_text())
        if selection['provenance']!=provenance(): raise RuntimeError('Frozen source/protocol changed')
        if args.phase=='test' and not (OUT/'flowers102/confirmation/summary.json').exists():
            raise RuntimeError('Complete frozen confirmation before test')
        seeds=cfg['selection_seeds'] if args.phase=='confirmation' else cfg['final_seeds']
        run(args.dataset,args.phase,[r['config'] for r in selection['selected']],seeds,save_predictions=True)


if __name__=='__main__': main()
