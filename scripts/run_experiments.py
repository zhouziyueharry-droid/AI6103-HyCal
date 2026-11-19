"""Validation-first execution. A frozen selection is required for test scoring."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import statistics

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import torch
import hycal_core as core


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')
    os.replace(temporary,path)


def load_inputs():
    cfg=json.loads((ROOT/'configs/part_a.json').read_text())
    feature_path=ROOT/'artifacts/features/flowers102/features.pt'
    cache=torch.load(feature_path,map_location='cpu',weights_only=False)
    records=json.loads((ROOT/'data/manifests/flowers102/records.json').read_text())
    support=json.loads((ROOT/'data/manifests/flowers102/support_indices.json').read_text())
    assert cache['sample_ids']==[r['sample_id'] for r in records]
    assert cache['metadata']['manifest_sha256']==sha(ROOT/'data/manifests/flowers102/records.json')
    return cfg,cache,records,support


def prepare_stats(cache,records,support,seed,shot,device):
    ids=support['indices'][str(seed)][str(shot)]
    lookup={r['sample_id']:i for i,r in enumerate(records)}
    bylabel={c['label']:[] for c in cache['classes']}
    for sid in ids:
        row=lookup[sid]
        assert records[row]['split']=='train'
        bylabel[records[row]['label']].append(row)
    positions=[bylabel[c['label']] for c in cache['classes']]
    assert all(len(p)==shot for p in positions)
    x=cache['image_features'][torch.tensor(positions)].to(device)
    text=cache['text_features'].to(device)
    fused=core.normalize_embeddings(x+text[:,None,:])
    prototypes,cov=core.class_statistics(fused)
    # Match the upstream background order (support manifest order), not label order.
    label_to_candidate={c['label']:i for i,c in enumerate(cache['classes'])}
    bg_images=cache['image_features'][[lookup[sid] for sid in ids]].to(device)
    bg_text=text[[label_to_candidate[records[lookup[sid]]['label']] for sid in ids]]
    bg_fused=core.normalize_embeddings(bg_images+bg_text)
    bg_mu=bg_fused.mean(0)
    centered=bg_fused-bg_mu
    bg_cov=centered.T@centered/(len(bg_fused)-1)
    return prototypes,cov,text,(bg_mu,bg_cov)


def invert(cov):
    # Use the same torch.pinverse algorithm/rcond as the official implementation.
    return torch.stack([torch.pinverse(matrix) for matrix in cov])


def evaluate_one(cfg,cache,records,support,seed,shot,rho,split,methods,output_dir,device,lambda_reg=None):
    lambda_reg=cfg['lambda_reg'] if lambda_reg is None else lambda_reg
    torch.cuda.reset_peak_memory_stats() if device=='cuda' else None
    started=time.perf_counter()
    prototypes,cov,text,bg=prepare_stats(cache,records,support,seed,shot,device)
    precision=invert(core.regularize_covariance(cov,rho,lambda_reg))
    bg_mu,bg_cov=bg
    bg_precision=torch.pinverse(bg_cov+cfg['lambda_reg']*torch.eye(bg_cov.shape[0],device=device))
    if device=='cuda': torch.cuda.synchronize()
    stats_seconds=time.perf_counter()-started
    idx=[i for i,r in enumerate(records) if r['split']==split]
    images=cache['image_features'][idx].to(device)
    labels=cache['labels'][idx]
    candidate_labels=torch.tensor([c['label'] for c in cache['classes']])
    started=time.perf_counter()
    cos,md=core.score_features(images,text,prototypes,precision,cfg['batch_size_scoring'],(bg_mu,bg_precision))
    if device=='cuda': torch.cuda.synchronize()
    score_seconds=time.perf_counter()-started
    counts=torch.full((len(text),),shot)
    raw_diagnostics={}
    output=[]
    for method in methods:
        scores=core.fuse_scores(cos,md,counts,method,cfg['n_min_p'],cfg['scale'])
        predictions=candidate_labels[scores.argmax(1)]
        metrics=core.classification_metrics(predictions,labels,len(text))
        tag=f'{split}_k{shot}_s{seed}_{method}_rho{rho:g}_lambda{lambda_reg:g}'
        row={'tag':tag,'split':split,'seed':seed,'shot':shot,'method':method,'rho':rho,'lambda_reg':lambda_reg,
             'top1':metrics['top1'],'macro':metrics['macro'],'n':metrics['n'],
             'stats_seconds':stats_seconds,'scoring_seconds':score_seconds,
             'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else 0,
             'md_enabled_fraction':0.0 if method=='C0' else (1.0 if method=='MD' or (shot>=10 if method=='H0' else shot>1) else 0.0)}
        save_json(output_dir/(tag+'.json'),{'summary':row,'metrics':metrics})
        with (output_dir/(tag+'_predictions.csv')).open('w',newline='',encoding='utf-8') as f:
            writer=csv.writer(f)
            writer.writerow(['sample_id','true_label','pred_label','correct','cosine_pred','md_pred'])
            cp=candidate_labels[cos.argmax(1)].tolist();mp=candidate_labels[md.argmax(1)].tolist()
            writer.writerows((records[i]['sample_id'],int(y),int(p),int(y==p),a,b)
                            for i,y,p,a,b in zip(idx,labels,predictions,cp,mp))
        raw_diagnostics[method]=scores
        output.append(row)
        print(json.dumps(row),flush=True)
    if shot<cfg['n_min_p'] and 'C0' in raw_diagnostics and 'H0' in raw_diagnostics:
        assert torch.equal(raw_diagnostics['C0'],raw_diagnostics['H0'])
    return output


def select(validation_rows,cfg,provenance):
    means={}
    for rho in cfg['rho_candidates']:
        rows=[r for r in validation_rows if r['method']=='H2' and r['rho']==rho]
        if sorted(r['seed'] for r in rows)!=sorted(cfg['seeds']):
            raise ValueError('Selection requires all pre-registered seeds for each rho')
        means[str(rho)]=statistics.mean(r['macro'] for r in rows)
    chosen=max(cfg['rho_candidates'],key=lambda rho:(means[str(rho)],-rho))
    return {'selected_rho':chosen,'validation_macro_means':means,'selection_metric':cfg['selection_metric'],
            'tie_break':cfg['selection_tie_break'],'seeds':cfg['seeds'],'selection_shot':5,
            'frozen_at_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
            'provenance':provenance,'scope':'frozen for all test seeds and shots'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=['validation','test','sensitivity','replication'],required=True)
    parser.add_argument('--shots',type=int,nargs='+',default=[5])
    args=parser.parse_args()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    cfg,cache,records,support=load_inputs()
    device='cuda' if torch.cuda.is_available() else 'cpu'
    out=ROOT/'results/part_a'/args.phase
    out.mkdir(parents=True,exist_ok=True)
    selection_path=ROOT/'results/part_a/selection.json'
    provenance={'config_sha256':sha(ROOT/'configs/part_a.json'),
                'support_sha256':sha(ROOT/'data/manifests/flowers102/support_indices.json'),
                'feature_sha256':sha(ROOT/'artifacts/features/flowers102/features.pt'),
                'core_sha256':sha(ROOT/'src/hycal_core.py')}
    rows=[]
    if args.phase=='validation':
        if selection_path.exists(): raise RuntimeError('Selection already frozen; do not overwrite')
        for seed in cfg['seeds']:
            for rho in cfg['rho_candidates']:
                methods=['C0','H0','H1','H2','MD'] if rho==0 else ['H2','MD']
                rows+=evaluate_one(cfg,cache,records,support,seed,5,rho,'val',methods,out,device)
        save_json(out/'summary.json',rows)
        selection=select(rows,cfg,provenance)
        save_json(selection_path,selection)
        print('SELECTION FROZEN '+json.dumps(selection),flush=True)
    elif args.phase in {'test','replication'}:
        selection=json.loads(selection_path.read_text())
        if selection['provenance']!=provenance: raise RuntimeError('Frozen provenance changed')
        reference=ROOT/'results/part_a/reference_verification.json'
        if not reference.exists() or not json.loads(reference.read_text())['passed']:
            raise RuntimeError('Official reference verification must pass before test scoring')
        rho=selection['selected_rho']
        lambda_control_path=ROOT/'results/part_a/selection_lambda.json'
        lambda_control=json.loads(lambda_control_path.read_text()) if lambda_control_path.exists() else None
        if lambda_control is not None and lambda_control['provenance']!=provenance:
            raise RuntimeError('Lambda control provenance changed')
        if args.phase=='replication':
            support=json.loads((ROOT/'data/manifests/flowers102/replication_support_indices.json').read_text())
            cfg=dict(cfg,seeds=support['seeds'])
            if args.shots!=[5]: raise ValueError('Replication pre-registered for 5-shot only')
        for shot in args.shots:
            for seed in cfg['seeds']:
                rows+=evaluate_one(cfg,cache,records,support,seed,shot,0.,'test',
                                   ['C0','H0','H1','MD']+(['H2'] if rho==0 else []),out,device)
                if rho!=0:
                    rows+=evaluate_one(cfg,cache,records,support,seed,shot,rho,'test',['H2'],out,device)
                if lambda_control is not None and shot==5 and lambda_control['selected_lambda']!=cfg['lambda_reg']:
                    rows+=evaluate_one(cfg,cache,records,support,seed,shot,0.,'test',['H1'],out,device,lambda_control['selected_lambda'])
        save_json(out/('summary_k'+'_'.join(map(str,args.shots))+'.json'),rows)
    else:
        # Predefined equal-budget lambda tuning control, using validation only.
        if (ROOT/'results/part_a/selection_lambda.json').exists():
            raise RuntimeError('Lambda control already frozen')
        for seed in cfg['seeds']:
            for lam in [1e-5,1e-4,1e-3]:
                rows+=evaluate_one(cfg,cache,records,support,seed,5,0.,'val',['H1'],out,device,lam)
        save_json(out/'summary.json',rows)
        means={str(lam):statistics.mean(r['macro'] for r in rows if r['lambda_reg']==lam)
               for lam in [1e-5,1e-4,1e-3]}
        chosen=max([1e-5,1e-4,1e-3],key=lambda lam:(means[str(lam)],-lam))
        save_json(ROOT/'results/part_a/selection_lambda.json',
                  {'selected_lambda':chosen,'validation_macro_means':means,'provenance':provenance,
                   'frozen_at_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
                   'purpose':'equal-sized three-candidate validation search control; no test-driven choice'})


if __name__=='__main__':
    main()
