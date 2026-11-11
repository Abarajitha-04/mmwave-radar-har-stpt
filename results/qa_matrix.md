| Area | Check | Evidence | Status |
|---|---|---|---|
| C1 Literature | ≥15 papers 2022--26 (16 of 40 references) | paper/refs.bib | PASS |
| C1 Gap | Research gap and contributions stated | paper Sec. I--II | PASS |
| C2 Datasets | Both datasets parsed and used | results/eda_stats.json | PASS |
| C2 EDA | EDA and pre-processing figures | figures/eda_*, preproc_* | PASS |
| C2 Leakage | Automated audit: splits disjoint, train-only statistics | docs/leakage_audit.md | FAIL |
| C3 Model | Proposed model implemented, diagram | src/models.py, figures/arch.pdf | PASS |
| C4 Baselines | ≥3 baselines incl. SOTA (6 re-trained) | results/summary_radhar.csv | PASS |
| C4 Seeds | Proposed model, 3 seeds per dataset (3/3) | results/all_runs.csv | PASS |
| C4 Statistics | McNemar (Holm) and bootstrap CI | results/stats.json | PASS |
| C4 Cross-dataset | MiliPoint→RadHAR transfer | results/summary_transfer.csv | FAIL |
| C4 Ablation | Component ablation | results/summary_ablation.csv | PASS |
| Robustness | 7 perturbation types, 3 severities, both datasets | results/robustness_*.csv | FAIL |
| Training | Loss/accuracy/LR logged every epoch for every run | results/runs/*/history.csv | PASS |
| Metrics | Accuracy, macro P/R/F1, per-class, top-3, latency, size | metrics.json | PASS |
| C5 XAI | Attention, occlusion, saliency, t-SNE | results/xai.json | PASS |
| Checkpoint | Exported model + inference script | checkpoints/, src/inference.py | PASS |
| Tests | pytest (23 passed, 0 failed) | results/pytest.log | PASS |
| C7 Demo | Streamlit demo, headless test (28 steps, 0 failed) | results/apptest.json | PASS |
| C7 Demo video | Screen recording of the live demo | figures/demo/demo.mp4 | PASS |
| Config | Hyper-parameters in a config file | configs/default.yaml | PASS |
| Documentation | README, DEVLOG, reproduction script | README.md, docs/ | PASS |
| No placeholders | No pending text in paper or report sources | grep PENDINGTEXT | FAIL |
| C6 Paper | Compiled / built and inspected page by page | paper/main.pdf | PASS |
| Report | Compiled / built and inspected page by page | report/report.pdf | FAIL |
| C7 Slides | Compiled / built and inspected page by page | presentation/Team10_mmWave_HAR.pptx | PASS |