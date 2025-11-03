#!/usr/bin/env bash

set -euo pipefail

BASELINE_URL="https://aistages-api-public-prod.s3.amazonaws.com/app/Competitions/000372/data/code.tar.gz"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASELINE_DIR="${ROOT_DIR}/baseline"
ARCHIVE_PATH="${BASELINE_DIR}/baseline_code.tar.gz"

mkdir -p "${BASELINE_DIR}"

echo "[+] Downloading baseline code to ${ARCHIVE_PATH}"
if [ -f "${ARCHIVE_PATH}" ]; then
  echo "[=] Archive already exists, skipping download."
else
  curl -L "${BASELINE_URL}" -o "${ARCHIVE_PATH}"
fi

echo "[+] Extracting archive into ${BASELINE_DIR}"
tar -xzf "${ARCHIVE_PATH}" -C "${BASELINE_DIR}"

echo "[+] Baseline download complete."

