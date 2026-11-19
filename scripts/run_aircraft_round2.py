"""Additional preregistered blind-domain transfer, using frozen Flowers choices."""
import json
import run_round2 as base
import run_propagation_round2 as prop


def main():
    base.torch.set_num_threads(8)
    base.torch.backends.cuda.matmul.allow_tf32=False;base.torch.backends.cudnn.allow_tf32=False
    a=json.loads((base.OUT/'selection.json').read_text());b=json.loads((prop.OUT/'selection.json').read_text())
    assert a['provenance']==base.provenance() and b['provenance']==prop.provenance()
    assert (base.OUT/'joint_flowers_dtd/test/summary.json').exists()
    cfg=json.loads(base.CONFIG.read_text())
    files={p:base.sha(base.ROOT/p) for p in ['scripts/run_aircraft_round2.py','reports/part_a_round2/aircraft_transfer_protocol.md',
           'results/part_a_round2/selection.json','results/part_a_round2/propagation/selection.json']}
    p=base.OUT/'aircraft_transfer_execution_lock.json'
    if p.exists():assert json.loads(p.read_text())['files']==files
    else:base.save(p,{'created_utc':base.now(),'files':files,'policy':'No Aircraft tuning. Both dataset phase locks additionally bind inputs.'})
    base.run('aircraft','test',[r['config'] for r in a['selected']],cfg['final_seeds'],True)
    prop.run('aircraft','test',[r['config'] for r in b['selected']],cfg['final_seeds'])


if __name__=='__main__':main()
