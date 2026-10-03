#!/usr/bin/env bash
set -euo pipefail
for notebook in notebooks/*.ipynb; do
  echo "Notebook: ${notebook}"
done
echo "Execute notebooks after placing verified data in data/raw/."