#!/usr/bin/env bash
set -e

# Check mode
if [ "$1" = "-profile" ] || [ "$1" = "-p" ]; then
    echo "Running with profiling..."
    py-spy python ./Training.py --profile
else
    echo "Running training..."
    python ./Training.py
fi