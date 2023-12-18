#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail

if [[ -f $3 ]]; then
  touch "$3"
  exit 0
else
  echo "$(tput bold)$(tput setaf 9)Manual $1: '$2' -> '$3'$(tput sgr0)"
  exit 1
fi
