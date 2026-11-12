#!/usr/bin/env bash
# Regenerate every derived artefact from results/ in dependency order (run from the repository root).
set -e
export PPTX_SKILL=${PPTX_SKILL:-/root/.claude/skills/synced/47d48c49-8041-4a71-842a-a21ff6df46fd_4dc27009-b8dc-4f0e-ab5d-3e20e24d663b/pptx}
python3 scripts/analyze.py > results/analyze.log 2>&1
python3 scripts/leakage_audit.py > results/leakage_audit.log 2>&1
python3 scripts/make_numbers.py
python3 scripts/deck_data.py
python3 scripts/make_readme.py
(cd presentation && node build_deck.js && python3 "$PPTX_SKILL/scripts/office/validate.py" Team10_mmWave_HAR.pptx)
python3 scripts/export_notes.py
latex() { (cd "$1" && pdflatex -interaction=nonstopmode "$2.tex" >/dev/null; bibtex "$2" >/dev/null 2>&1; \
           pdflatex -interaction=nonstopmode "$2.tex" >/dev/null; pdflatex -interaction=nonstopmode "$2.tex" > /dev/null; \
           echo "$1: $(pdfinfo $2.pdf | grep Pages) errors: $(grep -c '^!' $2.log) undefined: $(grep -c 'undefined' $2.log)") || true; }
latex paper main
python3 scripts/make_qa.py
latex report report
python3 scripts/make_qa.py >/dev/null   # second pass: report.pdf now exists
latex report report
