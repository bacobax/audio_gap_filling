#!/usr/bin/env python3
"""
Main entry point for the refactored AI training framework.

This script demonstrates how to use the new modular framework to train
MAE models with clean, maintainable code.
"""

import sys
import os
import argparse

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.factory import TrainingPipeline


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Run training pipeline")
    parser.add_argument(
        "--config",
        default="default",
        help="Name of the configuration file (without .yaml) or path to a YAML file"
    )
    args = parser.parse_args()
    config_name = args.config
    if config_name.endswith('.yaml'):
        config_path = config_name
    else:
        config_path = os.path.join('configs', f"{config_name}.yaml")

    if not os.path.exists(config_path):
        print(f"Configuration file not found: {config_path}")
        return
    
    try:
        # Create and run the training pipeline
        pipeline = TrainingPipeline(config_path)
        history = pipeline.run()
        
        print("Training completed!")
        if history:
            print(f"Final training loss: {history.get('train_losses', [0])[-1]:.6f}")
            if history.get('val_losses'):
                print(f"Final validation loss: {history['val_losses'][-1]:.6f}")
    
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 