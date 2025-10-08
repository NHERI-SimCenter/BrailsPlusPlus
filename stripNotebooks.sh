]#!/usr/bin/env bash
# strip-notebooks.sh
# Traverse current directory tree and strip outputs from all .ipynb files.

set -euo pipefail

# If you want a dry run, run: DRY_RUN=1 ./strip-notebooks.sh
DRY_RUN="${DRY_RUN:-0}"

# Find notebooks, excluding typical virtualenv/conda/cache dirs
find . -type f -name "*.ipynb" \
  -not -path "*/.venv/*" \
  -not -path "*/venv/*" \
  -not -path "*/.conda/*" \
  -not -path "*/.ipynb_checkpoints/*" \
  -print0 |
while IFS= read -r -d '' nb; do
    echo "Stripping: $nb"
    # -q = quiet, -f = force write even if file not under git
    jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace "$nb"
    # nbstripout -q -f "$nb"
done

echo "Done."
