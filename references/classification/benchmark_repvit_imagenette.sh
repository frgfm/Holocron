#!/usr/bin/env bash
# Copyright (C) 2019-2026, François-Guillaume Fernandez.
# Licensed under the Apache License 2.0; see LICENSE.
set -euo pipefail

if (( $# < 2 )); then
    printf 'Usage: bash %s DATA OUTPUT_DIR [training options]\n' "$0" >&2
    exit 2
fi
data_path=$1
output_dir=$2
shift 2
shopt -s nullglob dotglob
entries=("$output_dir"/*)
if (( ${#entries[@]} )); then
    printf '%s\n' 'Use a fresh output directory to preserve previous results' >&2
    exit 2
fi
mkdir -p "$output_dir"
for arch in repvit_m0_9 repvit_m1_0 repvit_m1_1 mobileone_s2; do
    uv run --no-sync python -m references.classification.train "$data_path" \
        --amp --workers 4 --threads 4 "$@" \
        --arch "$arch" --output-file "$output_dir/$arch.pth" --benchmark
done
