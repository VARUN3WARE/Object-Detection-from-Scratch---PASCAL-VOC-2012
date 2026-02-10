#!/usr/bin/env python3
"""
Quick start example for object detection pipeline.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print(f"\nSuccess: {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nError: {description} failed")
        print(f"Exit code: {e.returncode}")
        return False


def main():
    """Run the complete pipeline."""
    print("\n" + "="*60)
    print("Object Detection Pipeline - Quick Start")
    print("="*60)
    
    # Check if data exists
    data_path = Path("./data")
    if not data_path.exists():
        print("\nError: Data directory not found.")
        print("Please ensure PASCAL VOC 2012 dataset is in ./data/")
        return
    
    # Step 1: Training
    if not run_command(
        [sys.executable, "train.py"],
        "Step 1: Training Model"
    ):
        print("\nTraining failed. Please check the logs.")
        return
    
    # Step 2: Evaluation
    if not run_command(
        [sys.executable, "evaluate.py"],
        "Step 2: Evaluating Model"
    ):
        print("\nEvaluation failed. Please check the logs.")
        return
    
    # Step 3: Inference
    if not run_command(
        [sys.executable, "inference.py", "--test-set"],
        "Step 3: Running Inference on Test Set"
    ):
        print("\nInference failed. Please check the logs.")
        return
    
    print("\n" + "="*60)
    print("Pipeline Complete!")
    print("="*60)
    print("\nOutputs:")
    print("  - Model weights: ./weights_from_scratch/best_model.pth")
    print("  - Training curves: ./weights_from_scratch/training_curves.png")
    print("  - Evaluation report: ./weights_from_scratch/evaluation_report.json")
    print("  - Inference results: ./inference_results/test_set_predictions.png")
    print("\n")


if __name__ == "__main__":
    main()
