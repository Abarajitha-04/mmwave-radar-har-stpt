#!/usr/bin/env bash
# Reproduce every experiment in the paper (CPU-only; ~8 h on 2 cores).
set -e
# 1. data
git clone --depth 1 https://github.com/nesl/RadHAR ext/RadHAR
git clone --depth 1 https://github.com/yizzfz/MiliPoint ext/MiliPoint   # for id.json
#    download MiliPoint_data.zip (Google Drive link in the MiliPoint README) and unzip into ext/milipoint_raw/
python3 src/radhar_parse.py ext/RadHAR/Data data/processed/radhar_frames.npy
# 2. EDA
python3 scripts/eda.py
# 3. all training runs (2 parallel workers)
python3 scripts/runner.py scripts/jobs_main.txt 2
# 4. tables, statistics, figures, explainability, robustness
python3 scripts/analyze.py
python3 scripts/xai_robust.py robust
python3 scripts/xai_robust.py xai
python3 scripts/make_numbers.py
