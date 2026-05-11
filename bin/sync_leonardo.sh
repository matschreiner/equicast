#!/bin/bash
REMOTE="leonardo:/leonardo/home/userexternal/jschrei1/equicast"

if [ $# -eq 0 ]; then
    dirs=(jobs experiments)
else
    dirs=("$@")
fi

for dir in "${dirs[@]}"; do
    rsync -avpr "$REMOTE/$dir/" "$dir/"
done
