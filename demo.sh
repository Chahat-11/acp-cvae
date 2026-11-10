#!/bin/bash
# Live demo: generate new anticancer peptides and score them.
# Run with:  bash demo.sh
set -e
cd "$(dirname "$0")"

echo ""
echo "=============================================="
echo " GENERATING NEW ANTICANCER PEPTIDE CANDIDATES"
echo "=============================================="
echo ""
echo "The model has never seen these peptides. It is"
echo "inventing them from a random latent vector."
echo ""
sleep 2

PYTHONPATH=src .venv/bin/python src/generate.py \
    --n 600 --threshold 0.99 --checkpoint cvae_ar.pt

echo ""
echo "=============================================="
echo " SCORING THEM WITH AntiCP 2.0"
echo "=============================================="
echo ""
echo "AntiCP 2.0 is a published predictor (Raghava lab,"
echo "2021). We did not train it. It judges our output."
echo ""
sleep 2

PYTHONPATH=src .venv/bin/python src/score_with_anticp.py

echo ""
echo "Full ranked list: outputs/ranked_candidates.csv"
echo ""
