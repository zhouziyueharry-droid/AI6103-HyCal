"""Check SPD Cholesky implementation against upstream-style pseudoinverse.

Uses only 64 development queries and seed0 support; no confirmation/test data.
"""
import json
import run_round2 as base
import run_propagation_round2 as prop
import propagation_core as pc


def main():
    torch=base.torch;torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cache,records,support,query_idx=base.inputs('flowers102','dev')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    ctx=base.context(cache,records,support,query_idx[:64],0,device)
    selection=json.loads((prop.OUT/'selection.json').read_text())
    rows=[]
    for entry in selection['selected'][:3]:
        cfg=entry['config']
        prior=pc.propagated_prior(ctx['raw'],ctx['texts'],cfg['mode'])[0]
        covariance=(1-cfg['alpha'])*ctx['cov']+cfg['alpha']*prior
        covariance+=cfg['ridge']*torch.eye(covariance.shape[-1],device=device)
        cholesky=pc.positive_precision(covariance)
        reference=base.invert(covariance)
        rel=float((cholesky-reference).norm()/reference.norm())
        cos,a=base.r2.quadratic_scores(ctx['query'],ctx['texts'],ctx['proto'],cholesky)
        _,b=base.r2.quadratic_scores(ctx['query'],ctx['texts'],ctx['proto'],reference)
        cos=base.core.normalize_scores(cos)
        sa=(1-cfg['weight'])*cos+cfg['weight']*base.core.normalize_scores(a)
        sb=(1-cfg['weight'])*cos+cfg['weight']*base.core.normalize_scores(b)
        error=float((sa-sb).abs().max());agreement=float((sa.argmax(1)==sb.argmax(1)).float().mean())
        passed=rel<=1e-3 and error<=1e-4 and agreement==1.
        rows.append({'config':cfg,'precision_relative_frobenius_error':rel,'normalized_score_max_absolute_error':error,
                     'prediction_agreement':agreement,'passed':passed})
    result={'created_utc':base.now(),'scope':'seed0,64 Flowers development queries,3 selected covariance families',
            'thresholds':{'precision_relative':1e-3,'normalized_score_absolute':1e-4,'prediction_agreement':1.},
            'provenance':prop.provenance(),'passed':all(r['passed'] for r in rows),'rows':rows}
    base.save(base.OUT/'propagation_precision_verification.json',result)
    print(json.dumps(result,indent=2),flush=True)
    if not result['passed']:raise RuntimeError('Cholesky reference check failed')


if __name__=='__main__':main()
