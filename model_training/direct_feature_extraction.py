import os
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.feature_extraction import image as skimage
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, hog
from skimage import filters, measure
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class DirectFeatureExtractor:
    def __init__(self, test_dir='data/splits/test', output_dir='extracted_features_direct'):
        """
        Initialize direct feature extractor
        
        Args:
            test_dir: Directory containing normal/stenosis subdirectories
            output_dir: Directory to save extracted features
        """
        self.test_dir = test_dir
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Feature extraction parameters
        self.hog_orientations = 8
        self.hog_pixels_per_cell = (16, 16)
        self.hog_cells_per_block = (2, 2)
        self.lbp_radius = 3
        self.lbp_n_points = 8
        
        # Get image paths
        self.image_paths = self._get_image_paths()
        
    def _get_image_paths(self):
        """Get all image paths from normal and stenosis directories"""
        print("Scanning for images...")
        
        image_paths = []
        
        # Check for normal and stenosis directories
        normal_dir = os.path.join(self.test_dir, 'normal')
        stenosis_dir = os.path.join(self.test_dir, 'stenosis')
        
        if os.path.exists(normal_dir):
            print(f"Found normal directory: {normal_dir}")
            for filename in os.listdir(normal_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    image_paths.append({
                        'path': os.path.join(normal_dir, filename),
                        'filename': filename,
                        'class': 'normal',
                        'class_label': 0
                    })
        
        if os.path.exists(stenosis_dir):
            print(f"Found stenosis directory: {stenosis_dir}")
            for filename in os.listdir(stenosis_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    image_paths.append({
                        'path': os.path.join(stenosis_dir, filename),
                        'filename': filename,
                        'class': 'stenosis',
                        'class_label': 1
                    })
        
        print(f"Found {len(image_paths)} images total")
        print(f"  Normal: {len([p for p in image_paths if p['class'] == 'normal'])}")
        print(f"  Stenosis: {len([p for p in image_paths if p['class'] == 'stenosis'])}")
        
        return image_paths
    
    def _load_image(self, image_path):
        """Load image from file"""
        try:
            # Load with PIL first to handle different formats
            pil_image = Image.open(image_path)
            # Convert to RGB if needed
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            # Convert to numpy array
            image = np.array(pil_image)
            return image
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return None
    
    def _extract_geometric_features(self, image):
        """Extract geometric features from entire image"""
        if image.size == 0:
            return {}
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image
        
        features = {}
        
        # Basic geometric features
        features['image_width'] = image.shape[1]
        features['image_height'] = image.shape[0]
        features['image_area'] = image.shape[0] * image.shape[1]
        features['aspect_ratio'] = image.shape[1] / image.shape[0] if image.shape[0] > 0 else 0
        features['perimeter'] = 2 * (image.shape[0] + image.shape[1])
        features['circularity'] = (4 * np.pi * features['image_area']) / (features['perimeter'] ** 2) if features['perimeter'] > 0 else 0
        
        # Shape analysis
        try:
            # Find contours
            _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                perimeter = cv2.arcLength(largest_contour, True)
                
                features['contour_area'] = area
                features['contour_perimeter'] = perimeter
                features['contour_circularity'] = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
                
                # Bounding rectangle
                x, y, w, h = cv2.boundingRect(largest_contour)
                features['bounding_box_area'] = w * h
                features['bounding_box_aspect_ratio'] = w / h if h > 0 else 0
                features['extent'] = area / features['bounding_box_area'] if features['bounding_box_area'] > 0 else 0
                
                # Additional shape features
                features['solidity'] = area / cv2.contourArea(cv2.convexHull(largest_contour)) if len(largest_contour) > 2 else 0
                
                # Calculate eccentricity
                if len(largest_contour) >= 5:
                    ellipse = cv2.fitEllipse(largest_contour)
                    features['eccentricity'] = np.sqrt(1 - (ellipse[1][1] / ellipse[1][0]) ** 2) if ellipse[1][0] > ellipse[1][1] else np.sqrt(1 - (ellipse[1][0] / ellipse[1][1]) ** 2)
                else:
                    features['eccentricity'] = 0
            else:
                features['contour_area'] = 0
                features['contour_perimeter'] = 0
                features['contour_circularity'] = 0
                features['bounding_box_area'] = 0
                features['bounding_box_aspect_ratio'] = 0
                features['extent'] = 0
                features['solidity'] = 0
                features['eccentricity'] = 0
        except:
            features['contour_area'] = 0
            features['contour_perimeter'] = 0
            features['contour_circularity'] = 0
            features['bounding_box_area'] = 0
            features['bounding_box_aspect_ratio'] = 0
            features['extent'] = 0
            features['solidity'] = 0
            features['eccentricity'] = 0
        
        return features
    
    def _extract_texture_features(self, image):
        """Extract texture features from entire image"""
        if image.size == 0:
            return {}
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image
        
        features = {}
        
        try:
            # Local Binary Pattern (LBP)
            lbp = local_binary_pattern(gray_image, self.lbp_n_points, self.lbp_radius, method='uniform')
            lbp_hist, _ = np.histogram(lbp, bins=np.arange(0, self.lbp_n_points + 3), density=True)
            features['lbp_entropy'] = -np.sum(lbp_hist * np.log2(lbp_hist + 1e-10))
            features['lbp_energy'] = np.sum(lbp_hist ** 2)
            features['lbp_uniformity'] = np.max(lbp_hist)
            
            # Gray Level Co-occurrence Matrix (GLCM)
            # Quantize image to reduce computation
            gray_quantized = (gray_image / 16).astype(np.uint8)
            glcm = graycomatrix(gray_quantized, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=16, symmetric=True, normed=True)
            
            # GLCM properties
            features['glcm_contrast'] = np.mean(graycoprops(glcm, 'contrast'))
            features['glcm_dissimilarity'] = np.mean(graycoprops(glcm, 'dissimilarity'))
            features['glcm_homogeneity'] = np.mean(graycoprops(glcm, 'homogeneity'))
            features['glcm_energy'] = np.mean(graycoprops(glcm, 'energy'))
            features['glcm_correlation'] = np.mean(graycoprops(glcm, 'correlation'))
            features['glcm_asm'] = np.mean(graycoprops(glcm, 'ASM'))
            
            # Additional texture features
            # Gabor filter responses
            from skimage.filters import gabor
            gabor_responses = []
            for frequency in [0.1, 0.3, 0.5]:
                for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
                    filt_real, filt_imag = gabor(gray_image, frequency=frequency, theta=theta)
                    gabor_responses.append(np.mean(filt_real))
                    gabor_responses.append(np.mean(filt_imag))
            
            features['gabor_mean'] = np.mean(gabor_responses)
            features['gabor_std'] = np.std(gabor_responses)
            
        except Exception as e:
            print(f"Error extracting texture features: {e}")
            features['lbp_entropy'] = 0
            features['lbp_energy'] = 0
            features['lbp_uniformity'] = 0
            features['glcm_contrast'] = 0
            features['glcm_dissimilarity'] = 0
            features['glcm_homogeneity'] = 0
            features['glcm_energy'] = 0
            features['glcm_correlation'] = 0
            features['glcm_asm'] = 0
            features['gabor_mean'] = 0
            features['gabor_std'] = 0
        
        return features
    
    def _extract_intensity_features(self, image):
        """Extract intensity-based features from entire image"""
        if image.size == 0:
            return {}
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image
        
        features = {}
        
        # Basic intensity statistics
        features['mean_intensity'] = np.mean(gray_image)
        features['std_intensity'] = np.std(gray_image)
        features['min_intensity'] = np.min(gray_image)
        features['max_intensity'] = np.max(gray_image)
        features['intensity_range'] = features['max_intensity'] - features['min_intensity']
        features['intensity_variance'] = np.var(gray_image)
        features['intensity_skewness'] = self._calculate_skewness(gray_image)
        features['intensity_kurtosis'] = self._calculate_kurtosis(gray_image)
        
        # Histogram features
        hist, _ = np.histogram(gray_image, bins=256, range=(0, 256), density=True)
        features['intensity_entropy'] = -np.sum(hist * np.log2(hist + 1e-10))
        features['intensity_energy'] = np.sum(hist ** 2)
        features['intensity_uniformity'] = np.max(hist)
        
        # Percentiles
        features['intensity_10th_percentile'] = np.percentile(gray_image, 10)
        features['intensity_25th_percentile'] = np.percentile(gray_image, 25)
        features['intensity_50th_percentile'] = np.percentile(gray_image, 50)
        features['intensity_75th_percentile'] = np.percentile(gray_image, 75)
        features['intensity_90th_percentile'] = np.percentile(gray_image, 90)
        
        # Color features (if RGB)
        if len(image.shape) == 3:
            # RGB channel statistics
            for i, color in enumerate(['red', 'green', 'blue']):
                channel = image[:, :, i]
                features[f'{color}_mean'] = np.mean(channel)
                features[f'{color}_std'] = np.std(channel)
                features[f'{color}_min'] = np.min(channel)
                features[f'{color}_max'] = np.max(channel)
            
            # Color ratios
            features['red_green_ratio'] = features['red_mean'] / features['green_mean'] if features['green_mean'] > 0 else 0
            features['red_blue_ratio'] = features['red_mean'] / features['blue_mean'] if features['blue_mean'] > 0 else 0
            features['green_blue_ratio'] = features['green_mean'] / features['blue_mean'] if features['blue_mean'] > 0 else 0
        
        return features
    
    def _calculate_skewness(self, data):
        """Calculate skewness of data"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data):
        """Calculate kurtosis of data"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _extract_hog_features(self, image):
        """Extract HOG features from entire image"""
        if image.size == 0:
            return {}
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image
        
        try:
            # Resize image to standard size for HOG
            resized_image = cv2.resize(gray_image, (128, 128))
            
            # Extract HOG features
            hog_features = hog(
                resized_image,
                orientations=self.hog_orientations,
                pixels_per_cell=self.hog_pixels_per_cell,
                cells_per_block=self.hog_cells_per_block,
                visualize=False
            )
            
            # Use HOG features as summary statistics
            features = {}
            features['hog_mean'] = np.mean(hog_features)
            features['hog_std'] = np.std(hog_features)
            features['hog_min'] = np.min(hog_features)
            features['hog_max'] = np.max(hog_features)
            features['hog_entropy'] = -np.sum(hog_features * np.log2(hog_features + 1e-10))
            features['hog_energy'] = np.sum(hog_features ** 2)
            
            # Additional HOG statistics
            features['hog_median'] = np.median(hog_features)
            features['hog_q25'] = np.percentile(hog_features, 25)
            features['hog_q75'] = np.percentile(hog_features, 75)
            
            return features
            
        except Exception as e:
            print(f"Error extracting HOG features: {e}")
            return {
                'hog_mean': 0, 'hog_std': 0, 'hog_min': 0, 'hog_max': 0,
                'hog_entropy': 0, 'hog_energy': 0, 'hog_median': 0,
                'hog_q25': 0, 'hog_q75': 0
            }
    
    def _extract_edge_features(self, image):
        """Extract edge-based features"""
        if image.size == 0:
            return {}
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image
        
        features = {}
        
        try:
            # Canny edge detection
            edges = cv2.Canny(gray_image, 50, 150)
            features['edge_density'] = np.sum(edges > 0) / edges.size
            features['edge_count'] = np.sum(edges > 0)
            
            # Sobel edges
            sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
            sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            features['sobel_mean'] = np.mean(sobel_magnitude)
            features['sobel_std'] = np.std(sobel_magnitude)
            features['sobel_max'] = np.max(sobel_magnitude)
            
            # Laplacian
            laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
            features['laplacian_mean'] = np.mean(laplacian)
            features['laplacian_std'] = np.std(laplacian)
            features['laplacian_variance'] = np.var(laplacian)
            
        except Exception as e:
            print(f"Error extracting edge features: {e}")
            features['edge_density'] = 0
            features['edge_count'] = 0
            features['sobel_mean'] = 0
            features['sobel_std'] = 0
            features['sobel_max'] = 0
            features['laplacian_mean'] = 0
            features['laplacian_std'] = 0
            features['laplacian_variance'] = 0
        
        return features
    
    def extract_features_for_image(self, image_info):
        """Extract all features for a single image"""
        # Load image
        image = self._load_image(image_info['path'])
        if image is None:
            return None
        
        # Extract all feature types
        features = {}
        features.update(self._extract_geometric_features(image))
        features.update(self._extract_texture_features(image))
        features.update(self._extract_intensity_features(image))
        features.update(self._extract_hog_features(image))
        features.update(self._extract_edge_features(image))
        
        # Add metadata
        features['filename'] = image_info['filename']
        features['class'] = image_info['class']
        features['class_label'] = image_info['class_label']
        
        return features
    
    def extract_all_features(self):
        """Extract features for all images"""
        print("Extracting features for all images...")
        
        all_features = []
        
        for image_info in tqdm(self.image_paths, desc="Extracting features"):
            # Extract features
            features = self.extract_features_for_image(image_info)
            
            if features is not None:
                all_features.append(features)
            else:
                print(f"Skipping {image_info['filename']} due to extraction errors")
        
        # Convert to DataFrame
        features_df = pd.DataFrame(all_features)
        
        # Save features
        output_file = os.path.join(self.output_dir, 'extracted_features_direct.csv')
        features_df.to_csv(output_file, index=False)
        print(f"Features saved to {output_file}")
        print(f"Extracted {len(features_df)} feature vectors with {len(features_df.columns)} features each")
        
        return features_df
    
    def analyze_features(self, features_df):
        """Analyze extracted features"""
        print("\n=== Feature Analysis ===")
        
        # Basic statistics
        print(f"Total samples: {len(features_df)}")
        print(f"Total features: {len(features_df.columns) - 3}")  # Exclude metadata columns
        
        # Class distribution
        class_counts = features_df['class'].value_counts()
        print(f"\nClass distribution:")
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count} ({count/len(features_df)*100:.1f}%)")
        
        # Feature statistics
        numeric_columns = features_df.select_dtypes(include=[np.number]).columns
        numeric_columns = [col for col in numeric_columns if col not in ['class_label']]
        
        print(f"\nFeature statistics:")
        print(features_df[numeric_columns].describe())
        
        # Class-wise feature analysis
        print(f"\n=== Class-wise Feature Analysis ===")
        for class_name in features_df['class'].unique():
            class_data = features_df[features_df['class'] == class_name]
            print(f"\n{class_name.upper()} class ({len(class_data)} samples):")
            
            # Show some key features
            key_features = ['mean_intensity', 'std_intensity', 'hog_mean', 'edge_density', 'glcm_contrast']
            for feature in key_features:
                if feature in numeric_columns:
                    mean_val = class_data[feature].mean()
                    std_val = class_data[feature].std()
                    print(f"  {feature}: {mean_val:.4f} ± {std_val:.4f}")
        
        # Save feature analysis
        analysis_file = os.path.join(self.output_dir, 'feature_analysis_direct.txt')
        with open(analysis_file, 'w') as f:
            f.write("=== Direct Feature Analysis ===\n")
            f.write(f"Total samples: {len(features_df)}\n")
            f.write(f"Total features: {len(numeric_columns)}\n\n")
            f.write("Class distribution:\n")
            for class_name, count in class_counts.items():
                f.write(f"  {class_name}: {count} ({count/len(features_df)*100:.1f}%)\n")
            f.write(f"\nFeature statistics:\n")
            f.write(features_df[numeric_columns].describe().to_string())
            
            f.write(f"\n\n=== Class-wise Feature Analysis ===\n")
            for class_name in features_df['class'].unique():
                class_data = features_df[features_df['class'] == class_name]
                f.write(f"\n{class_name.upper()} class ({len(class_data)} samples):\n")
                
                key_features = ['mean_intensity', 'std_intensity', 'hog_mean', 'edge_density', 'glcm_contrast']
                for feature in key_features:
                    if feature in numeric_columns:
                        mean_val = class_data[feature].mean()
                        std_val = class_data[feature].std()
                        f.write(f"  {feature}: {mean_val:.4f} ± {std_val:.4f}\n")
        
        print(f"Feature analysis saved to {analysis_file}")
        
        # Create feature importance plot
        self._plot_feature_importance(features_df, numeric_columns)
    
    def _plot_feature_importance(self, features_df, numeric_columns):
        """Plot feature importance based on class separation"""
        try:
            from sklearn.feature_selection import f_classif
            
            # Prepare data
            X = features_df[numeric_columns]
            y = features_df['class_label']
            
            # Calculate F-scores
            f_scores, p_values = f_classif(X, y)
            
            # Create feature importance DataFrame
            importance_df = pd.DataFrame({
                'feature': numeric_columns,
                'f_score': f_scores,
                'p_value': p_values
            })
            importance_df = importance_df.sort_values('f_score', ascending=False)
            
            # Plot top features
            plt.figure(figsize=(12, 8))
            top_features = importance_df.head(20)
            
            plt.subplot(2, 1, 1)
            plt.barh(range(len(top_features)), top_features['f_score'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('F-Score')
            plt.title('Top 20 Most Discriminative Features')
            plt.gca().invert_yaxis()
            
            plt.subplot(2, 1, 2)
            plt.barh(range(len(top_features)), -np.log10(top_features['p_value']))
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('-log10(p-value)')
            plt.title('Statistical Significance of Features')
            plt.gca().invert_yaxis()
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save importance data
            importance_df.to_csv(os.path.join(self.output_dir, 'feature_importance.csv'), index=False)
            print(f"Feature importance analysis saved to {self.output_dir}")
            
        except Exception as e:
            print(f"Could not create feature importance plot: {e}")

def main():
    parser = argparse.ArgumentParser(description='Extract features directly from images in normal/stenosis directories')
    parser.add_argument('--test_dir', type=str, default='data/splits/test', help='Path to test data directory')
    parser.add_argument('--output_dir', type=str, default='extracted_features_direct', help='Output directory for features')
    
    args = parser.parse_args()
    
    # Initialize feature extractor
    extractor = DirectFeatureExtractor(args.test_dir, args.output_dir)
    
    # Extract features
    features_df = extractor.extract_all_features()
    
    # Analyze features
    if features_df is not None and len(features_df) > 0:
        extractor.analyze_features(features_df)
        print(f"\nFeature extraction completed successfully!")
        print(f"Extracted {len(features_df)} feature vectors")
        print(f"Results saved in: {args.output_dir}")
    else:
        print("Feature extraction failed or no features were extracted.")

if __name__ == "__main__":
    main() 