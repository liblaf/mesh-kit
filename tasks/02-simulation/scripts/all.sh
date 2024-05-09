#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail

export DATA_DIR=$HOME/Documents/data
readarray -t ids < <(command ls "$DATA_DIR/CT")
for id in "${ids[@]}"; do
  echo "Processing Patient: $id"
  make TARGET_ID="$id" || true
done
