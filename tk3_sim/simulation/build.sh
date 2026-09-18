#!/bin/sh
set -eu
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
docker build --platform=linux/amd64 -t aerial-robotics-laboratory:ionic "$@" "$script_dir"
