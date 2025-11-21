#!/usr/bin/env python3
"""
Recall Sensitivity Analysis Runner Script

This script runs recall (sensitivity) analysis on the trained models
to understand how recall changes under different conditions.
"""

import os
import sys
from model_training.recall_sensitivity_analysis import RecallSensitivityAnalysis

def main():
    # Configuration
    test_dir = 'data/splits/test'
    results_dir = 'recall_sensitivity_results'
    
    # Check if test directory exists
    if not os.path.exists(test_dir):
        print(f"Error: Test directory not found at {test_dir}")
        print("Please ensure the test data is available")
        sys.exit(1)
    
    print("=== Recall (Sensitivity) Analysis ===")
    print(f"Test directory: {test_dir}")
    print(f"Results directory: {results_dir}")
    print()
    print("This analysis will examine how recall (sensitivity) changes under:")
    print("1. Different classification thresholds")
    print("2. Different noise levels")
    print("3. Different blur levels")
    print("4. Different brightness levels")
    print()
    
    # Initialize recall analyzer
    analyzer = RecallSensitivityAnalysis(test_dir, results_dir)
    
    # Generate comprehensive report
    analyzer.generate_recall_report()
    
    print(f"\nRecall analysis completed!")
    print(f"Results saved in: {results_dir}")

if __name__ == "__main__":
    main() 