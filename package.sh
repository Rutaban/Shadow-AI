#!/usr/bin/env bash
# Build the delivery archives and the GitHub upload tree.
set -euo pipefail
cd "$(dirname "$0")"
OUT="$(cd .. && pwd)/shadow_ai_deliverables"
rm -rf "$OUT"
mkdir -p "$OUT"

EXCLUDES=(--exclude '.venv' --exclude '__pycache__' --exclude '*.pyc'
          --exclude '.DS_Store' --include 'data/raw/DOWNLOAD.md' --exclude 'data/raw/*'
          --exclude 'data/flows/tls_record_traces.json'
          --exclude 'paper/preview.pdf')

# 1. GitHub upload tree: exactly the files to push.
GH="$OUT/github-upload/shadow-ai-exfiltration"
mkdir -p "$GH"
rsync -a "${EXCLUDES[@]}" ./ "$GH/"
cp paper/preview.pdf "$GH/paper/SOICT2026_Shadow_AI_Exfiltration.pdf"

# 2. Source archive of the same tree.
( cd "$OUT/github-upload" && zip -qr "$OUT/shadow-ai-exfiltration-source.zip" shadow-ai-exfiltration )

# 3. Paper deliverables on their own.
mkdir -p "$OUT/paper"
cp paper/SOICT2026_Shadow_AI_Exfiltration.docx "$OUT/paper/"
cp paper/preview.pdf "$OUT/paper/SOICT2026_Shadow_AI_Exfiltration.pdf"
mkdir -p "$OUT/paper/figures"
cp figures/*.png figures/*.pdf "$OUT/paper/figures/"
cp results/results.json results/scenario_profile.csv "$OUT/paper/"
( cd "$OUT" && zip -qr "$OUT/shadow-ai-exfiltration-paper.zip" paper )

# 4. Measured corpus on its own, for anyone who wants the data without the code.
mkdir -p "$OUT/dataset"
cp data/flows/ai_endpoint_flows.csv data/flows/ai_endpoint_sequences.npy "$OUT/dataset/"
cp results/scenario_profile.csv "$OUT/dataset/"
( cd "$OUT" && zip -qr "$OUT/shadow-ai-exfiltration-dataset.zip" dataset )

echo "wrote:"
find "$OUT" -maxdepth 2 -name '*.zip' -o -maxdepth 1 -type d | sort
