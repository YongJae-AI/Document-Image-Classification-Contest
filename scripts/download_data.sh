#!/usr/bin/env bash

set -euo pipefail

DATA_URL="https://aistages-api-public-prod.s3.amazonaws.com/app/Competitions/000372/data/data.tar.gz"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="${ROOT_DIR}/data/raw"
ARCHIVE_PATH="${RAW_DIR}/data.tar.gz"

mkdir -p "${RAW_DIR}"

echo "[+] Downloading competition data to ${ARCHIVE_PATH}"
if [ -f "${ARCHIVE_PATH}" ]; then
  echo "[=] Archive already exists, skipping download."
else
  curl -L "${DATA_URL}" -o "${ARCHIVE_PATH}"
fi

echo "[+] Extracting archive into ${RAW_DIR}"
tar -xzf "${ARCHIVE_PATH}" -C "${RAW_DIR}"

echo "[+] Data download complete."

