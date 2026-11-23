# Pooled Covariance Shrinkage for HyCal

Code, frozen protocols and results for the AI6103 (NTU) group project
**"Pooled Covariance Shrinkage for HyCal: Few-Shot Cross-Domain Class-Incremental Learning"**.
The final report is [`report/report.pdf`](report/report.pdf) (LaTeX source in [`report/`](report/)).

HyCal ([Lee et al., CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Lee_HyCal_A_Training-Free_Prototype_Calibration_Method_for_Cross-Discipline_Few-Shot_Class-Incremental_CVPR_2026_paper.html))
scores queries on frozen CLIP features with a calibrated mix of cosine similarity and a per-class
Mahalanobis distance. With K ≤ 10 support images in a 512-d fused space, each class covariance has
rank ≤ K. We shrink every class covariance toward the within-class covariance pooled over the
current session,

    S̃_c = (1 − α) S_c + α S_pool,   α = 0.75,

and leave prototypes, fusion, calibration and weighting unchanged. One scalar is added and no
images from earlier sessions are stored.

## Headline results

Test accuracy (%), mean ± std over support seeds 0, 1, 42. *Tuned HyCal* is the control: identical
to our method except α = 0.

| Dataset | K | Official HyCal script | Tuned HyCal (α = 0) | **Shrinkage (α = 0.75)** | Paired gain (pp) |
|---|---:|---:|---:|---:|---:|
| EuroSAT RGB | 5 | 71.37 ± 1.97 | 76.60 ± 1.63 | **79.53 ± 1.30** | +2.93 ± 1.24 |
| EuroSAT RGB | 10 | 75.67 ± 0.92 | 83.49 ± 1.01 | **86.40 ± 0.87** | +2.92 ± 0.25 |
| Galaxy10 DECaLS | 5 | 37.99 ± 1.81 | 37.99 ± 2.03 | **39.70 ± 2.04** | +1.71 ± 2.76 |
| Galaxy10 DECaLS | 10 | 40.00 ± 1.95 | 39.74 ± 0.70 | **43.74 ± 1.77** | +4.00 ± 1.08 |
| Flowers102 (held out) | 5 | – | 93.26 ± 0.19 | **95.41 ± 0.44** | +2.15 ± 0.31 |
| Flowers102 (held out) | 10 | – | 95.02 ± 0.35 | **97.17 ± 0.34** | +2.15 ± 0.36 |
| OrganAMNIST (held out) | 5 | – | 59.37 ± 0.53 | **64.30 ± 0.77** | +4.92 ± 0.59 |
| OrganAMNIST (held out) | 10 | – | 65.61 ± 1.44 | **71.63 ± 1.88** | +6.02 ± 1.18 |

Shrinkage improves 23 of the 24 paired 5/10-shot runs. The configuration was chosen on the EuroSAT
and Galaxy10 validation splits and applied to Flowers102 and OrganAMNIST without retuning. The
official `run_balanced.sh` (n_min = 20) disables Mahalanobis scoring for K ≤ 10, so it equals
cosine-only scoring.

## Repository layout

```
partC/                  Pooled covariance shrinkage (main method), EuroSAT / Galaxy10 / transfer experiments
  hycal_cached.py         HyCal + pooled shrinkage on cached CLIP features
  common.py               paths, dataset definitions, prompt templates
  indices/                frozen support indices (K ∈ {1,5,10}, seeds 0/1/42)
  results/                every number in the report (see the table below)
  member_b_reconstructed/ query-gated weighting score function (alternative design)
src/, scripts/, tests/  Projected shared visual covariance (alternative design) on Flowers102 / DTD / FGVC-Aircraft
configs/, data/manifests/, results/part_a*/, reports/part_a_round2/
                        frozen protocols, support manifests, predictions and statistics for src/ + scripts/
tools/handoff/          standard-library verifier that recomputes the Flowers/DTD/Aircraft metrics from saved predictions
external/HyCal-CIL/     official HyCal implementation (pinned commit 4558fb7) with bundled CLIP
report/                 final report: LaTeX source, figures (+ generating scripts) and PDF
```

Raw images, CLIP weights and feature caches are not tracked (`data/`, `artifacts/`, `partC/cache/`
are git-ignored).

## Where each number in the report comes from

| Report item | Source file(s) |
|---|---|
| Table 1 (datasets) | [`partC/common.py`](partC/common.py), [`data/manifests/flowers102/`](data/manifests/flowers102/), [`partC/results/paired_transfer/*/protocol.json`](partC/results/paired_transfer/) |
| Table 2 (configurations), validation α sweep | [`partC/results/param_search/selected.json`](partC/results/param_search/selected.json), [`validation.md`](partC/results/param_search/validation.md), [`val_grid.csv`](partC/results/param_search/val_grid.csv) |
| Table 3 (EuroSAT / Galaxy10) | [`partC/results/single_domain/*/summary.md`](partC/results/single_domain/), [`partC/results/param_search/ablation_shrink.md`](partC/results/param_search/ablation_shrink.md) |
| Table 4 (Flowers102 / OrganAMNIST transfer) | [`partC/results/paired_transfer/RESULTS.md`](partC/results/paired_transfer/RESULTS.md) and per-seed `runs.csv` |
| Table 5 (ablation), Figure 2 (α sensitivity) | [`partC/results/param_search/ablation_shrink.md`](partC/results/param_search/ablation_shrink.md) |
| Table 6 (alternative designs) | [`partC/results/reverse_eurosat/README.md`](partC/results/reverse_eurosat/README.md), [`member_b_reconstructed/RESULTS.md`](partC/results/reverse_eurosat/member_b_reconstructed/RESULTS.md); GDA comparisons in [`reports/part_a_round2/analysis/`](reports/part_a_round2/analysis/) |
| Figure 3 (OrganAMNIST per class), Flowers102 class counts | [`partC/results/paired_transfer/*/predictions.csv`](partC/results/paired_transfer/) |
| Table 7 and cross-domain analysis | [`partC/results/param_search/incremental.md`](partC/results/param_search/incremental.md), [`partC/results/incremental/`](partC/results/incremental/) |

## Setup

Python ≥ 3.9 with PyTorch and torchvision (install the build that matches your CUDA/CPU), then

```sh
pip install -r requirements.txt
```

[`partC/requirements-macos.txt`](partC/requirements-macos.txt) pins the exact environment used for
the EuroSAT / Galaxy10 runs. Encoding images needs the OpenAI CLIP ViT-B/16 weights; the bundled
CLIP in `external/HyCal-CIL/clip` downloads them on first use.

## Reproducing the main experiments

All commands run from the repository root.

**1. Check saved results without data or GPU** (standard library only):

```sh
python tools/handoff/verify_predictions.py
python -m unittest discover -s tools/handoff -p test_handoff.py
python -m pytest tests          # numerical contracts of src/ (one test also needs open_clip_torch)
```

**2. Prepare EuroSAT and Galaxy10.** Put EuroSAT RGB in `data/eurosat/2750/` with the CoOp split
`data/eurosat/split_zhou_EuroSAT.json`; put `Galaxy10_DECals.h5` in `data/Galaxy10_DECals/` and
convert it with `python partC/prepare_galaxy10.py` (stratified 80/10/10 split, output in
`data/Galaxy10_DECals/_galaxy_temp/`). The frozen
support indices are already in `partC/indices/` (regenerate with `partC/make_support_indices.py`).

**3. Cache CLIP features once per dataset**, written to `partC/cache/{dataset}_ViT-B-16.npz`:

```sh
python partC/extract_features.py --dataset eurosat
python partC/extract_features.py --dataset galaxy10
```

**4. Run the experiments** (CPU, under 0.5 s per run on cached features):

```sh
python partC/param_search.py      # validation grid → selected.json; test.md; incremental.md (Table 7)
python partC/ablation_shrink.py   # Table 5 and the α sweep (Figure 2)
python partC/run_single_domain.py --dataset eurosat    # zero-shot / cosine / MD / official baselines
python partC/run_single_domain.py --dataset galaxy10
python partC/run_incremental.py --shots 10 1 --all_orders   # per-session matrices in results/incremental/
```

**5. Held-out transfer and alternative designs** (inputs listed in each results README, placed in
`artifacts/paired_inputs/`):

```sh
python partC/run_paired_transfer.py flowers102
python partC/run_paired_transfer.py organamnist
python partC/run_reverse_eurosat.py --include-gate
python partC/run_member_b_eurosat.py
```

The projected-shared-covariance study on Flowers102 / DTD / FGVC-Aircraft is driven by
`scripts/prepare_*.py`, `scripts/cache_*.py` and `scripts/run_round2*.py` with the frozen protocols
in `configs/` and `data/manifests/`; `scripts/verify_official.py` checks the cached implementation
against the pinned upstream code (set `HYCAL_OFFICIAL_CHECKOUT` to a clean checkout of commit
4558fb7).

**Rebuilding the report**: compile `report/report.tex` with pdfLaTeX (e.g. Overleaf), or
`tectonic report/build_tectonic.tex` locally. Figures are regenerated with
`python report/figures/src/make_plots.py report/figures` and
`python report/figures/src/make_overview.py report/figures/src`.

## Acknowledgements and licences

`external/HyCal-CIL/` is the official HyCal code
([EJLEE5826/HyCal-CIL](https://github.com/EJLEE5826/HyCal-CIL/tree/4558fb7fa6aa389310c9e9dc7db7c2cbebc84a41)),
kept with its [LICENSE](external/HyCal-CIL/LICENSE) and the
[CLIP notice](external/HyCal-CIL/THIRD_PARTY_NOTICES.md). Datasets: EuroSAT, Galaxy10 DECaLS,
Oxford Flowers102, DTD, FGVC-Aircraft and MedMNIST v2 OrganAMNIST, each under its own licence.
