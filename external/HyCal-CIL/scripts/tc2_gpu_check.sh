#!/bin/bash
# Historical TC2 settings, copied from the user's successful AI6103 workflow.
# Verify current entitlement with mytcinfo before submitting.
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --job-name=hycal_gpu_check
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

set -eo pipefail
test -n "${SLURM_JOB_ID:-}" || { echo 'Submit with sbatch.' >&2; exit 2; }
cd "${HYCAL_ROOT:-$HOME/hycal/HyCal-CIL}"
module load anaconda
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-ai6103clean}"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
nvidia-smi
python - <<'PY'
import importlib.util
import json
import os
import platform
from pathlib import Path
import torch
import torchvision

report = {
    'job_id': os.environ['SLURM_JOB_ID'],
    'python': platform.python_version(),
    'torch': torch.__version__,
    'torchvision': torchvision.__version__,
    'cuda_build': torch.version.cuda,
    'cuda_available': torch.cuda.is_available(),
    'dependencies': {name: importlib.util.find_spec(name) is not None for name in
        ['open_clip', 'ftfy', 'regex', 'yaml', 'easydict', 'wandb', 'gdown',
         'h5py', 'timm', 'tqdm', 'medmnist', 'pandas', 'sklearn', 'scipy', 'seaborn']},
}
if report['cuda_available']:
    report['gpu'] = torch.cuda.get_device_name(0)
    x = torch.randn(512, 512, device='cuda')
    report['matrix_check_finite'] = bool(torch.isfinite(x @ x).all().item())
output = Path('results/preflight')
output.mkdir(parents=True, exist_ok=True)
path = output / f"gpu_check_{report['job_id']}.json"
path.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
if not report['cuda_available'] or not report.get('matrix_check_finite'):
    raise RuntimeError('GPU preflight failed; see ' + str(path))
print('GPU check passed. Missing dependencies must be installed before the CLIP smoke test.')
PY
