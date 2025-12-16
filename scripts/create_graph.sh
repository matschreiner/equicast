#!/usr/bin/env bash

set -e

if [ -z "$1" ]; then
    echo "Usage: create_graph <graph_name>"
    exit 1
fi

name="$1"

anemoi-graphs create "${name}.yaml" "${name}.pt" --overwrite
anemoi-graphs inspect "${name}.pt" "${name}"
