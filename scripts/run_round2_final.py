"""One frozen, resumable final run: two confirmations and three fixed endpoints."""
import json,time
import run_round2 as base
import run_propagation_round2 as propagation


def main():
    base.torch.set_num_threads(8)
    base.torch.backends.cuda.matmul.allow_tf32=False;base.torch.backends.cudnn.allow_tf32=False
    selection=json.loads((base.OUT/'selection.json').read_text())
    extra=json.loads((propagation.OUT/'selection.json').read_text())
    if selection['provenance']!=base.provenance():raise RuntimeError('Primary study changed')
    if extra['provenance']!=propagation.provenance():raise RuntimeError('Propagation study changed')
    precision_check=json.loads((base.OUT/'propagation_precision_verification.json').read_text())
    if not precision_check['passed'] or precision_check['provenance']!=propagation.provenance():
        raise RuntimeError('Propagation precision reference must pass with current inputs')
    files={**base.provenance(),**propagation.provenance()}
    for relative in ['scripts/run_round2_final.py','results/part_a_round2/propagation/selection.json',
                     'reports/part_a_round2/joint_protocol.md','scripts/run_joint_round2.py',
                     'scripts/verify_propagation_precision.py','results/part_a_round2/propagation_precision_verification.json']:
        files[relative]=base.sha(base.ROOT/relative)
    for dataset in ['flowers102','dtd','joint_flowers_dtd']:
        for name in ['records.json','classes.json','support_indices.json']:
            relative=f'data/manifests/{dataset}/{name}';files[relative]=base.sha(base.ROOT/relative)
        relative=f'artifacts/features/{dataset}/features.pt';files[relative]=base.sha(base.ROOT/relative)
    lock=base.OUT/'final_execution_lock.json'
    if lock.exists():
        if json.loads(lock.read_text())['files']!=files:raise RuntimeError('Final lock changed')
    else:
        if any(any(root.glob(f'*/{phase}/s*.json')) for root in [base.OUT,propagation.OUT] for phase in ['test','confirmation']):
            raise RuntimeError('Final outputs already present without a lock')
        candidates=[r for r in selection['selected'] if r['config']['family'] in ['RDA_global','RDA_semantic','RDA_transport']]+[extra['selected'][0]]
        geometric=[r for r in candidates if r['config']['family'] in ['RDA_transport','Propagation']]
        base.save(lock,{'locked_before_confirmation_and_test_utc':base.now(),'files':files,
                       'best_improvement_on_development':max(candidates,key=lambda r:r['macro']),
                       'best_geometry_hypothesis_on_development':max(geometric,key=lambda r:r['macro']),
                       'test_endpoints':['flowers102','dtd','joint_flowers_dtd'],
                       'scope':'Flowers is exploratory; DTD never tuned. No selection from any confirmation/test endpoint.'})
    cfg=json.loads(base.CONFIG.read_text());configs=[r['config'] for r in selection['selected']]
    extra_configs=[r['config'] for r in extra['selected']]
    stages=[('flowers102','confirmation',cfg['selection_seeds'])]
    stages += [(dataset,'test',cfg['final_seeds']) for dataset in ['flowers102','dtd','joint_flowers_dtd']]
    for dataset,phase,seeds in stages:
        for label,runner,cs in [('main',base.run,configs),('propagation',propagation.run,extra_configs)]:
            started=time.perf_counter();print(json.dumps({'event':'start','study':label,'dataset':dataset,'phase':phase,'utc':base.now()}),flush=True)
            if label=='main':runner(dataset,phase,cs,seeds,True)
            else:runner(dataset,phase,cs,seeds)
            print(json.dumps({'event':'done','study':label,'dataset':dataset,'phase':phase,'seconds':time.perf_counter()-started,'utc':base.now()}),flush=True)
    print('ALL FROZEN FINAL ENDPOINTS COMPLETE',flush=True)


if __name__=='__main__':main()
