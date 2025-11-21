#!/usr/bin/env python3
"""
Sensitivity Analysis Runner for ArteriAI Ensemble Model

This script runs comprehensive sensitivity analysis on the trained ensemble model
to understand its robustness to various perturbations and identify critical features.

Usage:
    python run_sensitivity_analysis.py
"""

import os
import sys
from model_training.sensitivity_analysis import EnsembleSensitivityAnalysis

def main():
    print("=" * 60)
    print("ARTERIAI ENSEMBLE MODEL SENSITIVITY ANALYSIS")
    print("=" * 60)
    
    # Check if models exist
    model_paths = [
        'models/vgg16_final_model.keras',
        'models/resnet50_final_model.keras', 
        'models/efficientnet_final_model.keras'
    ]
    
    missing_models = []
    for path in model_paths:
        if not os.path.exists(path):
            missing_models.append(path)
    
    if missing_models:
        print("❌ Missing trained models:")
        for path in missing_models:
            print(f"   - {path}")
        print("\nPlease train the models first using:")
        print("   python model_training/train_vgg16.py")
        print("   python model_training/train_resnet50.py") 
        print("   python model_training/train_efficientnet.py")
        sys.exit(1)
    
    # Check if test data exists
    test_dir = 'data/splits/test'
    if not os.path.exists(test_dir):
        print(f"❌ Test data directory not found: {test_dir}")
        print("Please ensure test data is available in the specified directory.")
        sys.exit(1)
    
    print("✅ All required files found!")
    print("\nStarting comprehensive sensitivity analysis...")
    
    try:
        # Initialize sensitivity analyzer
        analyzer = EnsembleSensitivityAnalysis(
            test_dir=test_dir,
            results_dir='sensitivity_analysis_results'
        )
        
        # Run comprehensive analysis
        report = analyzer.generate_sensitivity_report()
        
        print("\n" + "=" * 60)
        print("SENSITIVITY ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nResults saved to: sensitivity_analysis_results/")
        print("\nGenerated files:")
        print("  📊 noise_sensitivity_results.csv")
        print("  📊 blur_sensitivity_results.csv") 
        print("  📊 brightness_sensitivity_results.csv")
        print("  📊 model_agreement_matrix.csv")
        print("  📈 noise_sensitivity_analysis.png")
        print("  📈 blur_sensitivity_analysis.png")
        print("  📈 brightness_sensitivity_analysis.png")
        print("  📈 model_agreement_heatmap.png")
        print("  📄 sensitivity_analysis_report.txt")
        
        print("\n📋 Summary Report Preview:")
        print("-" * 40)
        lines = report.split('\n')[:20]  # Show first 20 lines
        for line in lines:
            print(line)
        print("... (see full report in sensitivity_analysis_report.txt)")
        
    except Exception as e:
        print(f"\n❌ Error during sensitivity analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 