#!/usr/bin/env python3
"""
Train script for the refactored MAE framework.

This script loads configuration, sets up the model, data, and trainer, and starts training.
"""
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.factory import TrainingPipeline

def main():
    config_path = "hparams.yaml"
    if not os.path.exists(config_path):
        print(f"Configuration file not found: {config_path}")
        return
    pipeline = TrainingPipeline(config_path)
    pipeline.setup_data()
    pipeline.setup_model()
    pipeline.setup_trainer()
    pipeline.train()

if __name__ == "__main__":
    main() 