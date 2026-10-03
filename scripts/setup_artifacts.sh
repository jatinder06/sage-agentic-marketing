#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DRIVE_FOLDER_URL="https://drive.google.com/drive/folders/1yeMPqOS6KL71qjDsJdwQjP-aIAuB82xR?usp=sharing"
DRIVE_FOLDER_ID="1yeMPqOS6KL71qjDsJdwQjP-aIAuB82xR"

command -v uv >/dev/null 2>&1 || {
    echo "uv is required. Install it from https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
}

mkdir -p "$ROOT/data/raw" "$ROOT/artifacts" "$ROOT/models"

# NOTE: Google Drive has no curl-friendly endpoint that zips an entire folder
# on demand — the "download as zip" button in the browser is a cookie/session
# feature of the Drive UI, not a stable API. Attempting it here (as a prior
# version of this script did) reliably fails, so we go straight to gdown's
# recursive folder download, which pulls each file individually and avoids
# the zip-splitting problem entirely.

download_with_retry() {
    local max_attempts=6
    local attempt=1
    local wait_seconds=15

    while (( attempt <= max_attempts )); do
        echo "Downloading shared folder from Google Drive (attempt $attempt/$max_attempts)..."
        if uvx --from gdown gdown \
            --folder "$DRIVE_FOLDER_URL" \
            --output "$ROOT" \
            --quiet; then
            return 0
        fi

        echo "Download attempt $attempt failed (likely rate-limited by Google Drive)." >&2
        if (( attempt < max_attempts )); then
            echo "Waiting ${wait_seconds}s before retrying..." >&2
            sleep "$wait_seconds"
            wait_seconds=$(( wait_seconds * 2 ))   # exponential backoff
        fi
        attempt=$(( attempt + 1 ))
    done

    echo "Failed to download the artifact folder after $max_attempts attempts." >&2
    echo "Google Drive may be rate-limiting this network/IP. Consider using rclone" >&2
    echo "with an authenticated Drive remote instead, which uses the official API" >&2
    echo "quota rather than the unauthenticated scraping path gdown relies on." >&2
    exit 1
}

download_with_retry

required_files=(
    "$ROOT/artifacts/distilbert_sentiment_production/final_model/sentiment_agent.py"
    "$ROOT/artifacts/distilbert_sentiment_production/final_model/config.json"
    "$ROOT/artifacts/distilbert_sentiment_production/final_model/model.safetensors"
    "$ROOT/artifacts/distilbert_sentiment_production/final_model/tokenizer.json"
    "$ROOT/artifacts/distilbert_sentiment_production/final_model/tokenizer_config.json"
    "$ROOT/artifacts/churn_xgboost_production/model/churn_xgboost_predictor.py"
)

missing_or_empty=0
for file in "${required_files[@]}"; do
    if [[ ! -f "$file" ]]; then
        echo "Required artifact is missing: $file" >&2
        missing_or_empty=1
    elif [[ ! -s "$file" ]]; then
        # Catches files left behind as 0 bytes by an interrupted/rate-limited download
        echo "Required artifact is present but empty (likely a truncated download): $file" >&2
        missing_or_empty=1
    fi
done

if [[ "$missing_or_empty" -ne 0 ]]; then
    echo "One or more artifacts failed verification. Re-run this script to retry," >&2
    echo "or delete the affected files/folders under '$ROOT' before retrying so" >&2
    echo "gdown re-downloads them cleanly." >&2
    exit 1
fi

echo "Artifact download complete. All required files verified present and non-empty."