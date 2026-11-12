# Privacy-Preserving Human Activity Recognition from mmWave Radar Point Clouds using Lightweight Point Transformers

**Team 10, 23I006 Deep Learning mini project (conference paper), PSG College of Technology**
Abarajitha S K (23I301) · G Aswath Rupan (23I321) · Indu Vadana M D (23I323) · Murugan S (23I338)

Camera-free activity recognition from sparse mmWave radar point clouds (no image of the person is ever formed).
**STPT-Lite** (Spatio-Temporal Point Transformer, Lite) applies self-attention among the points of short temporal
segments and then across segment tokens, adds an explicit radar motion descriptor, is distilled from a larger
teacher, trained with noise-robust augmentation and stored with INT8 weights.

| | RadHAR (5 activities) | MiliPoint (49 actions, **unseen people**) |
|---|---|---|
| STPT-Lite + KD + robust (ours) | **96.83 %** | **49.56 %** top-1, 78.03 % top-3 |
| Best re-implemented baseline | PointNet + noise-robust training: 97.38 % | PointNet: 32.57 % |
| Model size | 117.5 K params, 36.6 MFLOPs, 2.58 ms (1 CPU thread), 0.14 MB INT8 | |

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
  72.9 % of test frames inside training windows, and that
  RadHAR's official test recordings come from the same sessions as its training recordings.

## Results (mean ± std over seeds; generated from `results/`)

### MiliPoint, unseen subjects (49 actions)
| Model | Seeds | Accuracy (%) | Macro-F1 (%) | Top-3 (%) | Params | MFLOPs | CPU latency (ms) |
|---|---|---|---|---|---|---|---|
| PointNet | 3 | 32.57 ± 4.59 | 29.72 ± 4.50 | 59.50 | 81.9 K | 31.8 | 0.74 |
| PointNet+BiLSTM | 3 | 28.11 ± 4.06 | 28.09 ± 4.66 | 56.08 | 114.8 K | 6.6 | 0.74 |
| PointNet++ (SSG) | 3 | 8.07 ± 5.47 | 5.12 ± 5.76 | 21.44 | 424.7 K | 133.1 | 7.94 |
| STPT-Large (teacher) | 3 | 50.39 ± 1.39 | 50.84 ± 2.11 | 75.81 | 730.9 K | 139.5 | 3.65 |
| STPT-Lite | 3 | 45.79 ± 6.30 | 45.95 ± 7.05 | 73.90 | 120.4 K | 36.6 | 2.60 |
| STPT-Lite + KD | 3 | 48.55 ± 4.35 | 47.62 ± 6.08 | 76.35 | 120.4 K | 36.6 | 2.07 |
| **STPT-Lite + KD + robust (proposed)** | 3 | 49.56 ± 1.98 | 48.80 ± 1.68 | 78.03 | 120.4 K | 36.6 | 2.00 |
| PointNet + noise-robust training | 3 | 18.66 ± 8.83 | 15.67 ± 8.98 | 40.90 | 81.9 K | 31.8 | 0.78 |

### RadHAR, official test recordings
| Model | Seeds | Accuracy (%) | Macro-F1 (%) | Params | MFLOPs | CPU latency (ms) |
|---|---|---|---|---|---|---|
| Voxel TD-CNN+BiLSTM [RadHAR] | 2 | 95.01 ± 0.16 | 95.03 ± 0.17 | 45.9 K | 180.5 | 6.60 |
| PointNet | 3 | 96.31 ± 0.66 | 96.36 ± 0.64 | 76.2 K | 31.8 | 0.74 |
| PointNet+BiLSTM | 3 | 96.14 ± 0.23 | 96.16 ± 0.21 | 109.1 K | 6.6 | 0.83 |
| PointNet++ (SSG) | 3 | 97.34 ± 0.18 | 97.37 ± 0.19 | 413.4 K | 133.1 | 8.44 |
| DGCNN | 1 | 95.85 ± 0.00 | 95.89 ± 0.00 | 422.4 K | 137.9 | 6.95 |
| Point Transformer | 3 | 97.01 ± 0.25 | 97.04 ± 0.24 | 335.2 K | 264.4 | 7.98 |
| STPT-Large (teacher) | 3 | 96.44 ± 1.43 | 96.48 ± 1.42 | 725.3 K | 139.5 | 3.59 |
| STPT-Lite | 3 | 96.22 ± 1.43 | 96.27 ± 1.41 | 117.5 K | 36.6 | 2.14 |
| STPT-Lite + KD | 3 | 96.04 ± 0.78 | 96.08 ± 0.76 | 117.5 K | 36.6 | 2.08 |
| STPT-Tiny + KD | 3 | 96.86 ± 0.17 | 96.90 ± 0.16 | 30.7 K | 12.1 | 1.92 |
| **STPT-Lite + KD + robust (proposed)** | 3 | 96.83 ± 0.69 | 96.88 ± 0.68 | 117.5 K | 36.6 | 2.58 |
| PointNet + noise-robust training | 3 | 97.38 ± 0.11 | 97.40 ± 0.11 | 76.2 K | 31.8 | 0.73 |

### Evaluation protocol (STPT-Lite, MiliPoint)
| Split | Accuracy (%) |
|---|---|
| Subject-disjoint (ours) | 45.79 ± 6.30 (n=3) |
| Chronological (MiliPoint code) | 58.38 (n=1) |
| Random overlapping windows (leaky) | 79.48 (n=1) |

### Ablation (STPT-Lite without KD; change in test accuracy vs. the full model trained with the same seeds, points)
| Removed component | MiliPoint (unseen people) | RadHAR |
|---|---|---|
| − temporal Transformer | -11.9 (n=2) | +0.7 (n=2) |
| − intra-segment attention | -6.5 (n=2) | +0.9 (n=2) |
| − motion descriptor | +0.5 (n=2) | +1.7 (n=2) |
| − Doppler & intensity | – | +0.4 (n=2) |
| − clutter gating | – | +1.6 (n=2) |
| − all augmentation | – | +0.7 (n=2) |

RadHAR is saturated (all models 95–98 %), so its ablation differences lie inside the seed spread; MiliPoint is the informative benchmark.

### Cross-dataset transfer (MiliPoint pre-training → RadHAR, xyz input)
| RadHAR training data | From scratch | Pre-trained on MiliPoint |
|---|---|---|
| 10 % | 61.22 ± 0.58 (n=2) | 78.98 ± 4.34 (n=2) |
| 25 % | 86.10 ± 1.37 (n=2) | 87.45 ± 2.32 (n=2) |
| 100 % | 96.42 ± 1.69 (n=2) | 95.23 ± 2.02 (n=2) |

### Robustness (test accuracy %, identical perturbed windows for every model; full grid in `results/robustness_summary.csv`)
| Condition | MiliPoint: ours | PointNet | PointNet + robust | RadHAR: ours | PointNet | PointNet + robust |
|---|---|---|---|---|---|---|
| clean | 49.6 | 32.6 | 18.7 | 96.8 | 96.3 | 97.4 |
| jitter 10 cm | 22.8 | 14.3 | 10.6 | 95.0 | 93.3 | 94.2 |
| sparsity 12.5 % | 13.9 | 14.7 | 10.3 | 80.0 | 87.6 | 91.1 |
| outliers 20 % | 39.8 | 15.5 | 14.6 | 55.1 | 51.5 | 96.0 |
| frame loss 50 % | 39.0 | 24.7 | 15.8 | 95.6 | 93.7 | 95.2 |
| combined, severity 3 | 15.8 | 9.6 | 9.3 | 57.0 | 51.5 | 92.2 |

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
