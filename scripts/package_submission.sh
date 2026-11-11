#!/usr/bin/env bash
# Assemble final_submission/ from the built deliverables (run from the repository root).
set -euo pipefail
F=final_submission
rm -rf "$F" && mkdir -p "$F"
mk() { mkdir -p "$F/$1"; }

mk 01_source_code
cp -r src scripts configs tests requirements.txt README.md "$F/01_source_code/"
cp -r .streamlit "$F/01_source_code/" 2>/dev/null || true

mk 02_model_checkpoint
cp -r checkpoints/* "$F/02_model_checkpoint/"
cat > "$F/02_model_checkpoint/README.txt" <<'EOF'
Exported inference bundles of the proposed model (STPT-Lite + KD + noise-robust training).
  stpt_lite_radhar/     RadHAR, 5 activities (seed chosen on validation macro-F1)
  stpt_lite_milipoint/  MiliPoint, 49 actions, subject-disjoint (seed chosen on validation macro-F1)
Each folder: model.pt (state dict), bundle.json (classes, model arguments, pre-processing, train-only statistics),
metrics.json (test metrics of that run). Load with src/inference.py:
  python src/inference.py --bundle checkpoints/stpt_lite_radhar --input demo/samples/radhar_test_walk.txt
EOF

mk 03_demo_application
cp -r demo "$F/03_demo_application/"
cp figures/demo/*.png figures/demo/demo.mp4 "$F/03_demo_application/"
cat > "$F/03_demo_application/README.txt" <<'EOF'
Run from the repository root (01_source_code + 02_model_checkpoint as checkpoints/ + demo/):
  pip install -r requirements.txt streamlit plotly
  streamlit run demo/app.py
demo.mp4 is a screen recording of the running application; demo_*.png are screenshots of its four tabs.
Headless functional test: python scripts/apptest_demo.py  (results/apptest.json)
EOF

mk 04_research_paper
cp paper/main.pdf "$F/04_research_paper/Team10_STPT-Lite_IEEE_paper.pdf"
mkdir -p "$F/04_research_paper/latex_source"
cp -r paper/main.tex paper/refs.bib paper/sections paper/generated paper/IEEEtran.cls "$F/04_research_paper/latex_source/" 2>/dev/null || true

mk 05_technical_development_report
cp report/report.pdf "$F/05_technical_development_report/Team10_Technical_Development_and_Decision_Report.pdf"
mkdir -p "$F/05_technical_development_report/latex_source"
cp -r report/report.tex report/chapters "$F/05_technical_development_report/latex_source/"

mk 06_presentation
cp presentation/Team10_mmWave_HAR.pptx "$F/06_presentation/"
cp presentation/build_deck.js presentation/deck_data.json "$F/06_presentation/"

mk 07_speaker_notes
cp presentation/Team10_Speaker_Notes.docx presentation/Team10_Speaker_Notes.pdf "$F/07_speaker_notes/"

mk 08_results
cp results/*.csv results/*.json "$F/08_results/"
cp results/qa_matrix.md "$F/08_results/" 2>/dev/null || true

mk 09_figures
find figures -maxdepth 1 -type f \( -name '*.pdf' -o -name '*.png' \) -exec cp {} "$F/09_figures/" \;

mk 10_experiment_logs
for d in results/runs/*/; do t=$(basename "$d"); mkdir -p "$F/10_experiment_logs/runs/$t"; cp "$d"metrics.json "$d"history.csv "$F/10_experiment_logs/runs/$t/" 2>/dev/null || true; done
cp -r results/logs "$F/10_experiment_logs/train_stdout" 2>/dev/null || true
cp results/runner_main.log results/robust_*.log results/pytest.log results/analyze.log "$F/10_experiment_logs/" 2>/dev/null || true
cp scripts/jobs_main.txt scripts/jobs_optional.txt "$F/10_experiment_logs/"

mk 11_documentation
cp README.md docs/DEVLOG.md docs/leakage_audit.md "$F/11_documentation/"
cp results/qa_matrix.md "$F/11_documentation/QA_matrix.md" 2>/dev/null || true

(cd "$F" && find . -type f | sort > MANIFEST.txt)
rm -f Team10_final_submission.zip
zip -qr Team10_final_submission.zip "$F"
echo "files: $(wc -l < $F/MANIFEST.txt)  size: $(du -sh $F | cut -f1)  zip: $(du -h Team10_final_submission.zip | cut -f1)"
