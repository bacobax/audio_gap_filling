#!/bin/bash

## List of YAML configuration files to use for training.
# Modify the CONFIGS array below to specify which config files to run.
CONFIGS=(
    "mask050_l1_00_p1"
    "mask050_l1_00_p1_warmup"
    "mask050_l1_00_p05_warmup"
)

for config in "${CONFIGS[@]}"; do
    echo "Starting training with configuration: $config"
    python3 train.py --config "$config"

    if [ $? -ne 0 ]; then
        echo "Training failed for config: $config"
        exit 1
    fi

    echo "Finished training with configuration: $config"
    echo "----------------------------------------"
done