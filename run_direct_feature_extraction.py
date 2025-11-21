#!/usr/bin/env python3
"""
Direct Feature Extraction Runner Script

This script runs feature extraction directly on images in the normal/stenosis
folders without needing bounding box information.
"""

import os
import sys
from model_training.direct_feature_extraction import DirectFeatureExtractor

def main():
    # Configuration
    test_dir = 'data/splits/test'
    output_dir = 'extracted_features_direct'
    
    # Check if test directory exists
    if not os.path.exists(test_dir):
        print(f"Error: Test directory not found at {test_dir}")
        sys.exit(1)
    
    # Check for normal and stenosis subdirectories
    normal_dir = os.path.join(test_dir, 'normal')
    stenosis_dir = os.path.join(test_dir, 'stenosis')
    
    if not os.path.exists(normal_dir):
        print(f"Error: Normal directory not found at {normal_dir}")
        sys.exit(1)
    
    if not os.path.exists(stenosis_dir):
        print(f"Error: Stenosis directory not found at {stenosis_dir}")
        sys.exit(1)
    
    print("=== Direct Feature Extraction ===")
    print(f"Test directory: {test_dir}")
    print(f"Normal directory: {normal_dir}")
    print(f"Stenosis directory: {stenosis_dir}")
    print(f"Output directory: {output_dir}")
    print()
    print("This will extract features from entire images:")
    print("1. Geometric features (area, aspect ratio, contours, etc.)")
    print("2. Texture features (LBP, GLCM, Gabor filters)")
    print("3. Intensity features (statistics, histograms, color)")
    print("4. HOG features (Histogram of Oriented Gradients)")
    print("5. Edge features (Canny, Sobel, Laplacian)")
    print()
    
    # Initialize feature extractor
    extractor = DirectFeatureExtractor(test_dir, output_dir)
    
    # Extract features
    features_df = extractor.extract_all_features()
    
    # Analyze features
    if features_df is not None and len(features_df) > 0:
        extractor.analyze_features(features_df)
        print(f"\nDirect feature extraction completed successfully!")
        print(f"Extracted {len(features_df)} feature vectors")
        print(f"Results saved in: {output_dir}")
    else:
        print("Feature extraction failed or no features were extracted.")

if __name__ == "__main__":
    main() 