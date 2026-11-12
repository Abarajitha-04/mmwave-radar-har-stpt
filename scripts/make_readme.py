"""Write README.md with results tables generated from results/ (no hand-typed numbers)."""
import json, os, sys
import pandas as pd, numpy as np
sys.path.insert(0, 'scripts')
from analyze import load_all, agg, ORDER, NAMES

df = load_all()


def table(pfx, top3=False):
    t = agg(df, [pfx + o for o in ORDER])
    h = '| Model | Seeds | Accuracy (%) | Macro-F1 (%) |' + (' Top-3 (%) |' if top3 else '') + ' Params | MFLOPs | CPU latency (ms) |'
    sep = '|---|---|---|---|' + ('---|' if top3 else '') + '---|---|---|'
    rows = [h, sep]
    for _, r in t.iterrows():
        n = NAMES[r.group[len(pfx):]]
        if 'proposed' in n: n = f'**{n}**'
        rows.append(f'| {n} | {r.n} | {r.acc_mean:.2f} ± {r.acc_std:.2f} | {r.f1_mean:.2f} ± {r.f1_std:.2f} |' +
                    (f' {r.top3:.2f} |' if top3 else '') + f' {r.params / 1e3:.1f} K | {r.mflops:.1f} | {r.lat_ms:.2f} |')
    return '\n'.join(rows)


G = {r.group: r for _, r in agg(df, sorted(set(df.group))).iterrows()}
g = lambda k: (f'{G[k].acc_mean:.2f} ± {G[k].acc_std:.2f} (n={G[k].n})' if G[k].n > 1 else f'{G[k].acc_mean:.2f} (n=1)') if k in G else 'n/a'
def best(p):
    k = max((k for k in G if k.startswith(p) and k[3:] in NAMES and not k[3:].startswith('stpt')), key=lambda k: G[k].acc_mean)
    return f'{NAMES[k[3:]]}: {G[k].acc_mean:.2f} %'
import pandas as _pd
def paired(v, ref):
    a = df[df.group == v]
    if not len(a): return None
    f = df[(df.group == ref) & df.seed.isin(set(a.seed))]
    return a.acc.mean() - f.acc.mean(), len(a)
abl_rows = []
for k, n in [('notemporal', 'temporal Transformer'), ('noattn', 'intra-segment attention'), ('nomotion', 'motion descriptor'), ('xyz', 'Doppler & intensity'), ('nogate', 'clutter gating'), ('augnone', 'all augmentation')]:
    m, r = paired(f'mp_abl_{k}', 'mp_stpt_lite'), paired(f'rh_abl_{k}', 'rh_stpt_lite')
    abl_rows.append(f'| − {n} | ' + (f'{m[0]:+.1f} (n={m[1]})' if m else '–') + ' | ' + (f'{r[0]:+.1f} (n={r[1]})' if r else '–') + ' |')
rob = _pd.read_csv('results/robustness_summary.csv')
def rv(ds, mdl, c, l):
    x = rob[(rob.dataset == ds) & (rob.model == mdl) & (rob.condition == c) & (abs(rob.level - l) < 0.006)]
    return f'{x.acc.iloc[0]:.1f}' if len(x) else '–'
rob_rows = [f'| {lab} | ' + ' | '.join(rv(ds, m, c, l) for ds in ['milipoint', 'radhar'] for m in ['stpt_lite_kd_robust', 'pointnet', 'pointnet_robust']) + ' |' for lab, c, l in [('clean', 'clean', 0), ('jitter 10 cm', 'jitter', 0.1), ('sparsity 12.5 %', 'sparsity', 0.125), ('outliers 20 %', 'outliers', 0.2), ('frame loss 50 %', 'frameloss', 0.5), ('combined, severity 3', 'combined', 3)]]

ours = G.get('rh_stpt_lite_kd_robust'); oursm = G.get('mp_stpt_lite_kd_robust')
audit = json.load(open('results/leakage_audit.json')) if os.path.exists('results/leakage_audit.json') else {}
rnd = audit.get('milipoint', {}).get('random', {})

md = f"""# Privacy-Preserving Human Activity Recognition from mmWave Radar Point Clouds using Lightweight Point Transformers

**Team 10, 23I006 Deep Learning mini project (conference paper), PSG College of Technology**
Abarajitha S K (23I301) · G Aswath Rupan (23I321) · Indu Vadana M D (23I323) · Murugan S (23I338)

Camera-free activity recognition from sparse mmWave radar point clouds (no image of the person is ever formed).
**STPT-Lite** (Spatio-Temporal Point Transformer, Lite) applies self-attention among the points of short temporal
segments and then across segment tokens, adds an explicit radar motion descriptor, is distilled from a larger
teacher, trained with noise-robust augmentation and stored with INT8 weights.

| | RadHAR (5 activities) | MiliPoint (49 actions, **unseen people**) |
|---|---|---|
| STPT-Lite + KD + robust (ours) | **{ours.acc_mean:.2f} %** | **{oursm.acc_mean:.2f} %** top-1, {oursm.top3:.2f} % top-3 |
| Best re-implemented baseline | {best('rh_')} | {best('mp_')} |
| Model size | {ours.params / 1e3:.1f} K params, {ours.mflops:.1f} MFLOPs, {ours.lat_ms:.2f} ms (1 CPU thread), {ours.int8_mb:.2f} MB INT8 | |

Deliverables in this repository: source code, trained models (`checkpoints/`), Streamlit demo (`demo/`), IEEE paper
(`paper/main.pdf`), technical development report (`report/report.pdf`), presentation with speaker notes
(`presentation/`), all run logs and results (`results/`), leakage audit (`docs/leakage_audit.md`), development log
(`docs/DEVLOG.md`).

## Quick start

```bash
pip install -r requirements.txt streamlit plotly pytest
python -m pytest -q tests                       # 23 tests
streamlit run demo/app.py                       # interactive demo (http://localhost:8501)
python src/inference.py --bundle checkpoints/stpt_lite_radhar --input demo/samples/radhar_test_walk.txt
```

The demo loads the exported checkpoint, parses a raw RadHAR recording (or a MiliPoint `.npz` sample), shows the raw and
pre-processed point clouds, predicts per 2-s window with class probabilities and latency, and lets you inject
controlled radar noise. A screen recording is in `figures/demo/demo.mp4`.

## Data

| Dataset | Source | Licence | Notes |
|---|---|---|---|
| RadHAR / MMActivity | https://github.com/nesl/RadHAR | BSD-3-Clause | TI IWR1443, 30 Hz, 5 activities, official train/test recordings |
| MiliPoint | https://github.com/yizzfz/MiliPoint (data: Google Drive link in its README) | MIT | TI IWR1843, 24 Hz, 11 subjects, 49 actions |

```bash
git clone --depth 1 https://github.com/nesl/RadHAR ext/RadHAR
git clone --depth 1 https://github.com/yizzfz/MiliPoint ext/MiliPoint      # provides data/raw/id.json
# download MiliPoint_data.zip from the link in the MiliPoint README and unzip to ext/milipoint_raw/
python src/radhar_parse.py ext/RadHAR/Data data/processed/radhar_frames.npy
```
MiliPoint pickles are loaded with a restricted unpickler (`src/safe_pickle.py`) that only allows numpy arrays.

## Method

1. **Pre-processing** (`src/data_radhar.py`): drop antenna self-returns (x < 0.3 m); centre each 2-s window on the
   median of the *moving* returns (Doppler ≠ 0; static clutter has zero radial velocity); keep returns within 1.2 m
   of the person (removes ~50 % of returns, static multipath/clutter); split the window into 12 segments × 32 points.
2. **STPT-Lite** (`src/models.py`): shared point MLP → intra-segment multi-head self-attention with a learned 3-D
   relative-position bias → attentive + max pooling → + motion-descriptor embedding (centroid, its displacement, mean
   |Doppler|, log point count) → [CLS] + 2 Transformer blocks across the 12 segment tokens. No FPS / k-NN.
3. **Knowledge distillation** from STPT-Large (725 K params), **noise-robust augmentation** (`src/noise.py`),
   **INT8** per-channel weight quantisation.

## Leakage-free evaluation

* RadHAR: official test recordings; validation = 20 % of *training recordings* (≤ 55 windows each), not windows.
* MiliPoint: **subject-disjoint** (test subjects 8–10, validation 6–7).
* Checkpoints chosen on validation macro-F1; test evaluated once; the exported seed is chosen on validation.
* `scripts/leakage_audit.py` verifies split disjointness, duplicate recordings/frames, train-only statistics and
  checkpoint selection. It also shows that a random window split of MiliPoint places
  {100 * rnd.get('fraction_test_frames_seen_in_training', 0):.1f} % of test frames inside training windows, and that
  RadHAR's official test recordings come from the same sessions as its training recordings.

## Results (mean ± std over seeds; generated from `results/`)

### MiliPoint, unseen subjects (49 actions)
{table('mp_', top3=True)}

### RadHAR, official test recordings
{table('rh_')}

### Evaluation protocol (STPT-Lite, MiliPoint)
| Split | Accuracy (%) |
|---|---|
| Subject-disjoint (ours) | {g('mp_stpt_lite')} |
| Chronological (MiliPoint code) | {g('mpo_official_stpt_lite')} |
| Random overlapping windows (leaky) | {g('mpo_random_stpt_lite')} |

### Ablation (STPT-Lite without KD; change in test accuracy vs. the full model trained with the same seeds, points)
| Removed component | MiliPoint (unseen people) | RadHAR |
|---|---|---|
{chr(10).join(abl_rows)}

RadHAR is saturated (all models 95–98 %), so its ablation differences lie inside the seed spread; MiliPoint is the informative benchmark.

### Cross-dataset transfer (MiliPoint pre-training → RadHAR, xyz input)
| RadHAR training data | From scratch | Pre-trained on MiliPoint |
|---|---|---|
| 10 % | {g('rh_xyz_scratch_f0.1')} | {g('rh_xyz_mpinit_f0.1')} |
| 25 % | {g('rh_xyz_scratch_f0.25')} | {g('rh_xyz_mpinit_f0.25')} |
| 100 % | {g('rh_abl_xyz')} | {g('rh_xyz_mpinit_f1.0')} |

### Robustness (test accuracy %, identical perturbed windows for every model; full grid in `results/robustness_summary.csv`)
| Condition | MiliPoint: ours | PointNet | PointNet + robust | RadHAR: ours | PointNet | PointNet + robust |
|---|---|---|---|---|---|---|
{chr(10).join(rob_rows)}

Known weakness: on RadHAR, injected false detections hurt STPT-Lite much more than PointNet with the same robust training.

## Reproduce everything

```bash
python scripts/runner.py scripts/jobs_main.txt 2      # all training runs (CPU; resumable; skips finished runs)
python scripts/runner.py scripts/jobs_final.txt 2     # transfer + MiliPoint ablation
python scripts/analyze.py                              # tables, statistics, figures
python scripts/xai_robust.py robust radhar && python scripts/xai_robust.py robust milipoint
python scripts/xai_robust.py robustfig && python scripts/xai_robust.py xai
python scripts/leakage_audit.py && python scripts/make_numbers.py
python scripts/export_model.py rh_stpt_lite_kd_robust checkpoints/stpt_lite_radhar radhar
cd paper && latexmk -pdf main.tex
# or regenerate every table, figure, document and the deck in one go:
bash scripts/final_build.sh
```
Hyper-parameters: `configs/default.yaml`. Every run directory stores its arguments, configuration, history
(train/val loss and accuracy, learning rate), test predictions and checkpoint.

## Repository layout
```
src/          parsing, safe unpickling, datasets, models, noise, training, inference
scripts/      job runner + job lists, EDA, analysis, robustness/XAI, leakage audit, export, paper numbers
configs/      default.yaml
checkpoints/  exported inference bundles (model.pt + bundle.json + metrics.json)
demo/         Streamlit app + held-out test samples
tests/        pytest suite
results/      run logs (runs/*/metrics.json, history.csv), summaries, robustness, audit
figures/      all figures (incl. demo screenshots and video)
paper/        IEEE conference paper (LaTeX + PDF)
report/       technical development and decision report (LaTeX + PDF)
presentation/ PowerPoint deck with speaker notes
docs/         development log, leakage audit
```

## Limitations
RadHAR cannot test new users (same sessions, no subject IDs); MiliPoint offers xyz only; perturbations are synthetic;
latency measured on a desktop CPU thread; no fall labels; not clinically validated. See the paper and report.

## Citation of datasets
Singh et al., "RadHAR," ACM mmNets 2019 · Cui et al., "MiliPoint," NeurIPS Datasets & Benchmarks 2023.
"""
open('README.md', 'w').write(md)
print('README written', len(md))
