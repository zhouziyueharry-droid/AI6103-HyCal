"""Fixed 5-shot-selected settings transferred to 10-shot Flowers; four seeds."""
import csv,json,time
import run_round2 as base
import run_propagation_round2 as prop


def main():
    torch=base.torch;torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    selection=json.loads((base.OUT/'selection.json').read_text())
    extra=json.loads((prop.OUT/'selection.json').read_text())
    assert selection['provenance']==base.provenance() and extra['provenance']==prop.provenance()
    families=['H0','H2','GDA','RDA_global','RDA_transport']
    configs=[r['config'] for r in selection['selected'] if r['config']['family'] in families]
    configs += [r['config'] for r in extra['selected'][:3]]
    cache,records,support,query_idx=base.inputs('flowers102','test')
    lookup={r['sample_id']:i for i,r in enumerate(records)}
    device='cuda' if torch.cuda.is_available() else 'cpu'
    folder=base.OUT/'flowers102/test_10shot'
    files={**base.provenance(),**prop.provenance(),'scripts/run_round2_tenshot.py':base.sha(__file__),
           'results/part_a_round2/final_execution_lock.json':base.sha(base.OUT/'final_execution_lock.json')}
    lock=folder/'run_lock.json'
    if lock.exists():assert json.loads(lock.read_text())['files']==files
    else:base.save(lock,{'created_utc':base.now(),'files':files,'policy':'No new tuning; four original seeds and nested support10; H0/H2 retain official K-dependent weights.'})
    rows=[]
    for seed in [0,1,42,1993]:
        started=time.perf_counter();ids=support['indices'][str(seed)]['10']
        bylabel={c['label']:[] for c in cache['classes']}
        for sid in ids:
            i=lookup[sid];assert records[i]['split']=='train'
            bylabel[records[i]['label']].append(i)
        index=torch.tensor([bylabel[c['label']] for c in cache['classes']]);assert index.shape==(102,10)
        raw=cache['image_features'][index].to(device)
        prototype,cov,texts,(bg_mu,bg_cov)=base.prepare_stats(cache,records,support,seed,10,device)
        bg_p=torch.pinverse(bg_cov+1e-4*torch.eye(512,device=device))
        ctx={'raw':raw,'flat':raw.flatten(0,1),'support_labels':torch.arange(102,device=device).repeat_interleave(10),
             'texts':texts,'proto':prototype,'cov':cov,'background':(bg_mu,bg_p),
             'query':cache['image_features'][query_idx].to(device),'true':cache['labels'][query_idx],
             'candidate_labels':torch.tensor([c['label'] for c in cache['classes']]),'memo':{}}
        for cfg in configs:
            tag=f's{seed}_{base.identity(cfg)}';target=folder/(tag+'.json')
            if target.exists():rows.append(json.loads(target.read_text()));continue
            if cfg['family'] in ['H0','H2']:
                rho=0. if cfg['family']=='H0' else .5
                precision=base.invert(base.core.regularize_covariance(cov,rho,1e-4))
                cos,md=base.core.score_features(ctx['query'],texts,prototype,precision,128,ctx['background'])
                score=base.core.fuse_scores(cos,md,torch.full((102,),10),cfg['family'])
            elif cfg['family'].startswith('Propagation'):score=prop.scores(ctx,cfg)
            else:score=base.scores(ctx,cfg)
            pred=ctx['candidate_labels'][score.argmax(1)]
            metrics=base.core.classification_metrics(pred,ctx['true'],102)
            row={'tag':tag,'dataset':'flowers102','phase':'test_10shot','seed':seed,'shot':10,'config':cfg,
                 'macro':metrics['macro'],'top1':metrics['top1'],'n':len(query_idx),'computed_utc':base.now(),
                 'per_class':metrics['per_class'],'class_counts':metrics['class_counts']}
            base.save(target,row);rows.append(row)
            with (folder/(tag+'_predictions.csv')).open('w',newline='',encoding='utf-8') as f:
                w=csv.writer(f);w.writerow(['sample_id','true_label','pred_label','correct'])
                w.writerows((records[i]['sample_id'],int(y),int(p),int(y==p)) for i,y,p in zip(query_idx,ctx['true'],pred))
        print(json.dumps({'seed':seed,'shot':10,'configs':len(configs),'seconds':time.perf_counter()-started}),flush=True)
    base.save(folder/'summary.json',rows)


if __name__=='__main__':main()
