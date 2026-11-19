"""Preregistered four-step old-domain decomposition; no model selection."""
import csv,json,time
from pathlib import Path
import run_round2 as base
import run_propagation_round2 as prop
import propagation_core as pc


@base.torch.no_grad()
def raw_scores(ctx,cfg):
    torch=base.torch;core=base.core;r2=base.r2
    family=cfg['family'];t=ctx['texts'];q=ctx['query']
    if family in ['H0','H2']:
        rho=0. if family=='H0' else .5
        covariance=core.regularize_covariance(ctx['cov'],rho,1e-4)
        precision=base.invert(covariance)
        cosines,distances=[],[]
        for start in range(0,len(q),128):
            query=core.normalize_embeddings(q[start:start+128,None,:]+t[None,:,:])
            cosines.append(torch.einsum('bcd,cd->bc',query,ctx['proto']))
            diff=(query-ctx['proto'][None,:,:]).transpose(0,1).contiguous()
            md=-(torch.bmm(diff,precision)*diff).sum(-1).T
            bg_mu,bg_p=ctx['background'];bg_diff=query[:,0,:]-bg_mu
            md+=((bg_diff@bg_p)*bg_diff).sum(-1,keepdim=True)
            distances.append(md)
        return torch.cat(cosines),torch.cat(distances),0. if family=='H0' else 1/(1+__import__('math').exp(1))
    if family=='Propagation':
        prior=pc.propagated_prior(ctx['raw'],t,cfg['mode'])[0]
        covariance=(1-cfg['alpha'])*ctx['cov']+cfg['alpha']*prior
        covariance+=cfg['ridge']*torch.eye(covariance.shape[-1],device=q.device)
        precision=pc.positive_precision(covariance)
    else:
        if cfg.get('transport',False):
            residual=core.normalize_embeddings(ctx['raw']+t[:,None,:])-ctx['proto'][:,None,:]
            prior=r2.transported_prior(residual,ctx['proto'],t,cfg['neighbors'])
            covariance=(1-cfg['alpha'])*ctx['cov']+cfg['alpha']*prior
            covariance+=cfg['ridge']*torch.eye(covariance.shape[-1],device=q.device)
        else:covariance=r2.rda_covariance(ctx['cov'],t,cfg['alpha'],cfg['neighbors'],cfg['ridge'])
        precision=base.invert(covariance)
    cos,md=r2.quadratic_scores(q,t,ctx['proto'],precision)
    return cos,md,cfg['weight']


def load_predictions(folder,tag):
    with (folder/(tag+'_predictions.csv')).open(encoding='utf-8') as f:
        return {r['sample_id']:int(r['pred_label']) for r in csv.DictReader(f)}


def main():
    torch=base.torch;torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    selection=json.loads((base.OUT/'selection.json').read_text())
    extra=json.loads((prop.OUT/'selection.json').read_text())
    assert selection['provenance']==base.provenance()
    assert extra['provenance']==prop.provenance()
    families=['H0','H2','RDA_global','RDA_semantic','RDA_transport']
    configs=[r['config'] for r in selection['selected'] if r['config']['family'] in families]+[extra['selected'][0]['config']]
    cache,records,support,_=base.inputs('joint_flowers_dtd','test')
    indices=[i for i,r in enumerate(records) if r['split']=='test' and r['label']<102]
    device='cuda' if torch.cuda.is_available() else 'cpu'
    output=[]
    for seed in [0,1,42,1993]:
        started=time.perf_counter();ctx=base.context(cache,records,support,indices,seed,device)
        oldmask=(ctx['candidate_labels']<102).to(device)
        for cfg in configs:
            root=prop.OUT if cfg['family']=='Propagation' else base.OUT
            tag=f's{seed}_{base.identity(cfg)}'
            old_predictions=load_predictions(root/'flowers102/test',tag)
            final_predictions=load_predictions(root/'joint_flowers_dtd/test',tag)
            expected=torch.tensor([final_predictions[records[i]['sample_id']] for i in indices])
            a=torch.tensor([old_predictions[records[i]['sample_id'].removeprefix('flowers102/')] for i in indices])
            cos,md,w=raw_scores(ctx,cfg)
            oldscore=(1-w)*base.core.normalize_scores(cos[:,oldmask])+w*base.core.normalize_scores(md[:,oldmask])
            fullscore=(1-w)*base.core.normalize_scores(cos)+w*base.core.normalize_scores(md)
            oldlabels=ctx['candidate_labels'][oldmask.cpu()]
            b=oldlabels[oldscore.cpu().argmax(1)]
            c=oldlabels[fullscore[:,oldmask].cpu().argmax(1)]
            d=ctx['candidate_labels'][fullscore.cpu().argmax(1)]
            assert torch.equal(d,expected),f'Diagnostic D disagrees with frozen run: {seed} {cfg}'
            metrics={name:base.core.classification_metrics(p,ctx['true'],149) for name,p in [('A',a),('B',b),('C',c),('D',d)]}
            row={'seed':seed,'family':cfg['family'],'config':cfg,'stages':{k:{m:v[m] for m in ['macro','top1']} for k,v in metrics.items()},
                 'D_prediction_agreement':1.,'old_to_new_domain_fraction':float((d>=102).float().mean())}
            for metric in ['macro','top1']:
                row[metric+'_changes_pp']={f'{x}_to_{y}':metrics[y][metric]-metrics[x][metric] for x,y in [('A','B'),('B','C'),('C','D')]}
            output.append(row)
        base.save(base.ROOT/'reports/part_a_round2/domain_decomposition.json',{'scope':'Four support seeds; B/C restricted candidates are analysis-only; no reselection','rows':output})
        print(json.dumps({'seed':seed,'decomposition_methods':len(configs),'seconds':time.perf_counter()-started}),flush=True)
        del ctx


if __name__=='__main__':main()
