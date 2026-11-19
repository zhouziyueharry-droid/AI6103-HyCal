"""Bounded follow-on development study; never adapts from confirmation/test."""
import argparse,csv,json,statistics,time
import run_round2 as base
import propagation_core as pc
ROOT=base.ROOT
OUT=base.OUT/'propagation'
CONFIG=ROOT/'configs/propagation_round2.json'


def candidates():
    return [{'family':family,'mode':mode,'alpha':a,'ridge':l,'weight':w}
            for family,mode in [('Propagation','projected'),('Propagation_raw_tuned','raw'),('Propagation_scale_tuned','scale_only')]
            for a in [.5,1.] for l in [1e-5,1e-4,1e-3] for w in [.5,1.]]


def provenance():
    files=['src/propagation_core.py','scripts/run_propagation_round2.py','configs/propagation_round2.json',
           'reports/part_a_round2/propagation_protocol.md','results/part_a_round2/selection.json']
    return {**base.provenance(),**{p:base.sha(ROOT/p) for p in files}}


@base.torch.no_grad()
def scores(ctx,cfg):
    mode=cfg['mode'];memo=ctx['memo']
    if ('prop_prior',mode) not in memo:
        memo[('prop_prior',mode)]=pc.propagated_prior(ctx['raw'],ctx['texts'],mode)[0]
    prior=memo[('prop_prior',mode)]
    key=('prop_score',mode,cfg['alpha'],cfg['ridge'])
    if key not in memo:
        covariance=(1-cfg['alpha'])*ctx['cov']+cfg['alpha']*prior
        covariance=covariance+cfg['ridge']*base.torch.eye(covariance.shape[-1],device=covariance.device)
        precision=pc.positive_precision(covariance)
        cos,md=base.r2.quadratic_scores(ctx['query'],ctx['texts'],ctx['proto'],precision)
        memo[key]=(base.core.normalize_scores(cos).cpu(),base.core.normalize_scores(md).cpu())
    cos,md=memo[key]
    return (1-cfg['weight'])*cos+cfg['weight']*md


def run(dataset,phase,configs,seeds):
    cache,records,support,query_idx=base.inputs(dataset,phase)
    device='cuda' if base.torch.cuda.is_available() else 'cpu'
    folder=OUT/dataset/phase
    files={**provenance(),**{str(p.relative_to(ROOT)):base.sha(p) for p in
           [ROOT/f'data/manifests/{dataset}/{name}' for name in ['records.json','classes.json','support_indices.json']]
           +[ROOT/f'artifacts/features/{dataset}/features.pt']}}
    lock=folder/'run_lock.json'
    if lock.exists():
        if json.loads(lock.read_text())['files']!=files:raise RuntimeError('Run inputs changed')
    else:base.save(lock,{'created_utc':base.now(),'files':files})
    rows=[]
    for seed in seeds:
        started=time.perf_counter();ctx=base.context(cache,records,support,query_idx,seed,device)
        for cfg in configs:
            tag=f's{seed}_{base.identity(cfg)}';target=folder/(tag+'.json')
            if target.exists():
                r=json.loads(target.read_text());assert r['config']==cfg;rows.append(r);continue
            tick=time.perf_counter();sc=scores(ctx,cfg)
            assert base.torch.isfinite(sc).all()
            pred=ctx['candidate_labels'][sc.argmax(1)]
            met=base.core.classification_metrics(pred,ctx['true'],len(ctx['texts']))
            row={'tag':tag,'dataset':dataset,'phase':phase,'seed':seed,'shot':5,'config':cfg,
                 'macro':met['macro'],'top1':met['top1'],'n':len(query_idx),'computed_utc':base.now(),
                 'seconds_including_cache_reuse':time.perf_counter()-tick}
            if phase!='dev':
                row.update(per_class=met['per_class'],class_counts=met['class_counts'])
                with (folder/(tag+'_predictions.csv')).open('w',newline='',encoding='utf-8') as f:
                    w=csv.writer(f);w.writerow(['sample_id','true_label','pred_label','correct'])
                    w.writerows((records[i]['sample_id'],int(y),int(p),int(y==p)) for i,y,p in zip(query_idx,ctx['true'],pred))
            base.save(target,row);rows.append(row)
        print(json.dumps({'dataset':dataset,'phase':phase,'seed':seed,'configs':len(configs),'seconds':time.perf_counter()-started}),flush=True)
        del ctx
    base.save(folder/'summary.json',rows)
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['init','dev','confirmation','test'],required=True)
    p.add_argument('--dataset',choices=['flowers102','dtd','joint_flowers_dtd'],default='flowers102');args=p.parse_args()
    base.torch.set_num_threads(8);base.torch.backends.cuda.matmul.allow_tf32=False;base.torch.backends.cudnn.allow_tf32=False
    if args.phase=='init':
        if CONFIG.exists():raise RuntimeError('Already registered')
        base.save(CONFIG,{'registered_utc':base.now(),'candidates':candidates(),
                          'scope':'bounded follow-on development after observing initial development results, before confirmation/test',
                          'selection_seeds':[0,1,42,1993],'final_seeds':[0,1,42,1993]+list(range(2,18))})
        print('Registered 36 propagation candidates');return
    cfg=json.loads(CONFIG.read_text())
    if args.phase=='dev':
        if args.dataset!='flowers102':raise RuntimeError('Only Flowers development permitted')
        if (OUT/'selection.json').exists():raise RuntimeError('Selection already frozen')
        forbidden=[base.OUT/dataset/phase for dataset in ['flowers102','dtd','joint_flowers_dtd'] for phase in ['confirmation','test']]
        if any(folder.exists() and any(folder.glob('s*.json')) for folder in forbidden):
            raise RuntimeError('Confirmation/test already opened; no further propagation development permitted')
        rows=run('flowers102','dev',cfg['candidates'],cfg['selection_seeds'])
        ranking=[]
        for candidate in cfg['candidates']:
            group=[r for r in rows if r['config']==candidate]
            assert sorted(r['seed'] for r in group)==sorted(cfg['selection_seeds'])
            ranking.append({'config':candidate,'macro':statistics.mean(r['macro'] for r in group),'top1':statistics.mean(r['top1'] for r in group)})
        selected=[max((r for r in ranking if r['config']['family']==family),key=lambda r:r['macro'])
                  for family in ['Propagation','Propagation_raw_tuned','Propagation_scale_tuned']]
        best=selected[0]
        selected += [{'config':dict(best['config'],family='Propagation_raw',mode='raw')},
                  {'config':dict(best['config'],family='Propagation_scale',mode='scale_only')},
                  {'config':dict(best['config'],family='Propagation_local',alpha=0.)}]
        choice={'frozen_utc':base.now(),'provenance':provenance(),'selected':selected,'ranking':ranking}
        base.save(OUT/'selection.json',choice);print(json.dumps(choice['selected'],indent=2),flush=True)
    else:
        choice=json.loads((OUT/'selection.json').read_text())
        if choice['provenance']!=provenance():raise RuntimeError('Frozen inputs changed')
        if args.phase=='test' and not (OUT/'flowers102/confirmation/summary.json').exists():raise RuntimeError('Confirmation required')
        run(args.dataset,args.phase,[r['config'] for r in choice['selected']],
            cfg['selection_seeds'] if args.phase=='confirmation' else cfg['final_seeds'])


if __name__=='__main__':main()
