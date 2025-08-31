#!/bin/bash

## List of YAML configuration files to use for training.
# Modify the CONFIGS array below to specify which config files to run.

CONFIGS=(
    "diffusion"
)

for config in "${CONFIGS[@]}"; do
    echo "Starting training with configuration: $config"
    OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE python3 train.py --config "$config"

    if [ $? -ne 0 ]; then
        echo "Training failed for config: $config"
        exit 1
    fi

    echo "Finished training with configuration: $config"
    echo "----------------------------------------"
done