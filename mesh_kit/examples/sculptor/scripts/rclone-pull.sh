#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail

remote="${1:-"business:/data/CT"}"
data_dir="${2:-$HOME/Documents/CT}"

rclone copy "$remote" "$data_dir" --progress --checksum
