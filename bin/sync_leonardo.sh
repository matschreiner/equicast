#!/bin/bash
REMOTE="leonardo:/leonardo/home/userexternal/jschrei1/equicast"

EXCLUDE_FLAGS=()
DIRS=()

for arg in "$@"; do
    if [ "$arg" = "--no-checkpoints" ]; then
        EXCLUDE_FLAGS+=(--exclude="*.ckpt")
    else
        DIRS+=("$arg")
    fi
done

if [ ${#DIRS[@]} -eq 0 ]; then
    DIRS=(jobs experiments)
fi

for dir in "${DIRS[@]}"; do
    rsync -avpr --info=progress2 "${EXCLUDE_FLAGS[@]}" "$REMOTE/$dir/" "$dir/"
done
