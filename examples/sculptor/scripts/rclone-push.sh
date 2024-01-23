#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail

data_dir="${1:-$HOME/Documents/CT}"
remote="${2:-"business:/data/CT"}"

rclone sync "$data_dir" "$remote" --progress --checksum --delete-excluded \
  --include="*/{pre,post}/00-CT.nrrd" \
  --include="*/{pre,post}/02-{face,skull}-landmarks.txt" \
  --include="template/02-{face,skull}.ply" \
  --include="template/03-{face,skull}-landmarks.txt"
