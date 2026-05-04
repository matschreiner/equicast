#!/usr/bin/env bash

set -e

if [ -z "$1" ]; then
    echo "Usage: create_graph <path/to/graph.yaml>"
    exit 1
fi

yaml="$1"
dir="$(dirname "$yaml")"
base="$(basename "$yaml" .yaml)"

anemoi-graphs create "${yaml}" "${dir}/${base}.pt" --overwrite
anemoi-graphs inspect "${dir}/${base}.pt" "${dir}/${base}" 2>&1 | tee "${dir}/summary.txt"
