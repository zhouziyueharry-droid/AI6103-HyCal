"""Frozen two-stage candidate expansion: Flowers102 then Flowers102+DTD."""
import json
import run_round2 as run


def main():
    run.torch.set_num_threads(8)
    run.torch.backends.cuda.matmul.allow_tf32=False
    run.torch.backends.cudnn.allow_tf32=False
    selection=json.loads((run.OUT/'selection.json').read_text())
    if selection['provenance']!=run.provenance(): raise RuntimeError('Frozen source/protocol changed')
    if not (run.OUT/'flowers102/confirmation/summary.json').exists(): raise RuntimeError('Confirmation required')
    cfg=json.loads(run.CONFIG.read_text())
    run.run('joint_flowers_dtd','test',[r['config'] for r in selection['selected']],cfg['final_seeds'],True)


if __name__=='__main__':main()
