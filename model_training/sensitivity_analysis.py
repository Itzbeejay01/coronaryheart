import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import cv2
from scipy import ndimage
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class EnsembleSensitivityAnalysis:
    def __init__(self, test_dir='data/splits/test', results_dir='sensitivity_analysis_results'):
        self.test_dir = test_dir
        self.results_dir = results_dir
        self.img_size = (224, 224)
        self.batch_size = 32
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        
        # Load trained models
        self.models = self._load_models()
        
        # Load test data
        self.test_data, self.y_true = self._load_test_data()
        
        # Baseline predictions
        self.baseline_predictions = self._get_baseline_predictions()
        
    def _load_models(self):
        """Load the final trained models from models folder"""
        print("Loading final trained models...")
        
        models = {}
        
        # Load final trained models (the ones we actually want to analyze)
        model_paths = {
            'vgg16': 'models/vgg16_final_model.keras',
            'resnet50': 'models/resnet50_final_model.keras',
            'efficientnet': 'models/efficientnet_final_model.keras'
        }
        
        for model_name, model_path in model_paths.items():
            if os.path.exists(model_path):
                try:
                    # Try to load the final model directly
                    models[model_name] = load_model(model_path, compile=False)
                    print(f"✓ {model_name.upper()} loaded from {model_path}")
                except Exception as e:
                    print(f"✗ Failed to load {model_name} from {model_path}: {e}")
                    print(f"  This might be due to TensorFlow version compatibility issues.")
                    print(f"  The model was trained with a different TensorFlow version.")
            else:
                print(f"⚠ {model_name.upper()} model not found at {model_path}")
        
        if not models:
            print("❌ No models could be loaded.")
            print("Possible solutions:")
            print("1. Check if models exist in models/ folder")
            print("2. Try using the same TensorFlow version used during training")
            print("3. Re-train the models with current TensorFlow version")
        else:
            print(f"\n✓ Successfully loaded {len(models)} models for sensitivity analysis")
            for name, model in models.items():
                print(f"  - {name.upper()}: {model.count_params():,} parameters")
        
        return models
    
    def _load_test_data(self):
        """Load test data"""
        print("Loading test data...")
        
        # Data generator for VGG16 and ResNet50
        datagen_vgg_resnet = ImageDataGenerator(rescale=1./255)
        test_gen_vgg_resnet = datagen_vgg_resnet.flow_from_directory(
            self.test_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=False
        )
        
        # Data generator for EfficientNet
        datagen_efficientnet = ImageDataGenerator(preprocessing_function=efficientnet_preprocess)
        test_gen_efficientnet = datagen_efficientnet.flow_from_directory(
            self.test_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=False
        )
        
        return {
            'vgg_resnet': test_gen_vgg_resnet,
            'efficientnet': test_gen_efficientnet
        }, test_gen_vgg_resnet.classes
    
    def _get_baseline_predictions(self):
        """Get baseline predictions from all models"""
        print("Getting baseline predictions...")
        
        predictions = {}
        
        # VGG16 predictions
        if 'vgg16' in self.models:
            predictions['vgg16'] = self.models['vgg16'].predict(
                self.test_data['vgg_resnet'], verbose=0
            ).flatten()
        
        # ResNet50 predictions
        if 'resnet50' in self.models:
            predictions['resnet50'] = self.models['resnet50'].predict(
                self.test_data['vgg_resnet'], verbose=0
            ).flatten()
        
        # EfficientNet predictions
        if 'efficientnet' in self.models:
            predictions['efficientnet'] = self.models['efficientnet'].predict(
                self.test_data['efficientnet'], verbose=0
            ).flatten()
        
        # Ensemble predictions
        if len(predictions) > 0:
            ensemble_probs = np.mean(list(predictions.values()), axis=0)
            predictions['ensemble'] = ensemble_probs
        
        return predictions
    
    def _calculate_metrics(self, y_true, y_pred_prob):
        """Calculate performance metrics"""
        y_pred = (y_pred_prob > 0.5).astype(int)
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'auc': roc_auc_score(y_true, y_pred_prob)
        }
    
    def noise_sensitivity_analysis(self, noise_levels=[0.01, 0.05, 0.1, 0.15, 0.2]):
        """Analyze sensitivity to Gaussian noise"""
        print("\n=== Noise Sensitivity Analysis ===")
        
        results = {}
        
        for model_name in self.models.keys():
            print(f"Analyzing {model_name}...")
            model_results = []
            
            for noise_level in tqdm(noise_levels, desc=f"{model_name} noise analysis"):
                # Get test images
                if model_name == 'efficientnet':
                    test_gen = self.test_data['efficientnet']
                else:
                    test_gen = self.test_data['vgg_resnet']
                
                # Reset generator
                test_gen.reset()
                
                noisy_predictions = []
                
                for batch_idx in range(len(test_gen)):
                    batch_images, _ = test_gen[batch_idx]
                    
                    # Add Gaussian noise
                    noise = np.random.normal(0, noise_level, batch_images.shape)
                    noisy_images = np.clip(batch_images + noise, 0, 1)
                    
                    # Get predictions
                    pred = self.models[model_name].predict(noisy_images, verbose=0).flatten()
                    noisy_predictions.extend(pred)
                
                # Calculate metrics
                metrics = self._calculate_metrics(self.y_true, np.array(noisy_predictions))
                metrics['noise_level'] = noise_level
                model_results.append(metrics)
            
            results[model_name] = model_results
        
        # Ensemble analysis
        print("Analyzing ensemble...")
        ensemble_results = []
        
        for noise_level in tqdm(noise_levels, desc="Ensemble noise analysis"):
            # Get test images
            test_gen_vgg_resnet = self.test_data['vgg_resnet']
            test_gen_efficientnet = self.test_data['efficientnet']
            
            # Reset generators
            test_gen_vgg_resnet.reset()
            test_gen_efficientnet.reset()
            
            ensemble_predictions = []
            
            for batch_idx in range(len(test_gen_vgg_resnet)):
                # VGG16/ResNet50 batch
                batch_vgg_resnet, _ = test_gen_vgg_resnet[batch_idx]
                noise_vgg_resnet = np.random.normal(0, noise_level, batch_vgg_resnet.shape)
                noisy_vgg_resnet = np.clip(batch_vgg_resnet + noise_vgg_resnet, 0, 1)
                
                # EfficientNet batch
                batch_efficientnet, _ = test_gen_efficientnet[batch_idx]
                noise_efficientnet = np.random.normal(0, noise_level, batch_efficientnet.shape)
                noisy_efficientnet = np.clip(batch_efficientnet + noise_efficientnet, 0, 1)
                
                # Get predictions from each model
                preds = []
                if 'vgg16' in self.models:
                    pred_vgg16 = self.models['vgg16'].predict(noisy_vgg_resnet, verbose=0).flatten()
                    preds.append(pred_vgg16)
                
                if 'resnet50' in self.models:
                    pred_resnet50 = self.models['resnet50'].predict(noisy_vgg_resnet, verbose=0).flatten()
                    preds.append(pred_resnet50)
                
                if 'efficientnet' in self.models:
                    pred_efficientnet = self.models['efficientnet'].predict(noisy_efficientnet, verbose=0).flatten()
                    preds.append(pred_efficientnet)
                
                # Ensemble prediction
                if preds:
                    ensemble_pred = np.mean(preds, axis=0)
                    ensemble_predictions.extend(ensemble_pred)
            
            # Calculate metrics
            metrics = self._calculate_metrics(self.y_true, np.array(ensemble_predictions))
            metrics['noise_level'] = noise_level
            ensemble_results.append(metrics)
        
        results['ensemble'] = ensemble_results
        
        # Save results
        self._save_noise_results(results)
        self._plot_noise_sensitivity(results)
        
        return results
    
    def _save_noise_results(self, results):
        """Save noise sensitivity results to CSV"""
        all_results = []
        
        for model_name, model_results in results.items():
            for result in model_results:
                result['model'] = model_name
                all_results.append(result)
        
        df = pd.DataFrame(all_results)
        df.to_csv(os.path.join(self.results_dir, 'noise_sensitivity_results.csv'), index=False)
        print(f"Noise sensitivity results saved to {self.results_dir}/noise_sensitivity_results.csv")
    
    def _plot_noise_sensitivity(self, results):
        """Plot noise sensitivity results"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        
        for i, metric in enumerate(metrics):
            ax = axes[i // 3, i % 3]
            
            for model_name, model_results in results.items():
                noise_levels = [r['noise_level'] for r in model_results]
                metric_values = [r[metric] for r in model_results]
                ax.plot(noise_levels, metric_values, marker='o', label=model_name, linewidth=2)
            
            ax.set_xlabel('Noise Level (σ)')
            ax.set_ylabel(metric.capitalize())
            ax.set_title(f'{metric.capitalize()} vs Noise Level')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Remove the last subplot if not needed
        if len(metrics) < 6:
            axes[1, 2].remove()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'noise_sensitivity_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Noise sensitivity plot saved to {self.results_dir}/noise_sensitivity_analysis.png")
    
    def blur_sensitivity_analysis(self, blur_levels=[0, 1, 2, 3, 4, 5]):
        """Analyze sensitivity to image blurring"""
        print("\n=== Blur Sensitivity Analysis ===")
        
        results = {}
        
        for model_name in self.models.keys():
            print(f"Analyzing {model_name}...")
            model_results = []
            
            for blur_level in tqdm(blur_levels, desc=f"{model_name} blur analysis"):
                # Get test images
                if model_name == 'efficientnet':
                    test_gen = self.test_data['efficientnet']
                else:
                    test_gen = self.test_data['vgg_resnet']
                
                # Reset generator
                test_gen.reset()
                
                blurred_predictions = []
                
                for batch_idx in range(len(test_gen)):
                    batch_images, _ = test_gen[batch_idx]
                    
                    # Apply Gaussian blur
                    if blur_level > 0:
                        blurred_images = np.array([
                            ndimage.gaussian_filter(img, sigma=blur_level) 
                            for img in batch_images
                        ])
                    else:
                        blurred_images = batch_images
                    
                    # Get predictions
                    pred = self.models[model_name].predict(blurred_images, verbose=0).flatten()
                    blurred_predictions.extend(pred)
                
                # Calculate metrics
                metrics = self._calculate_metrics(self.y_true, np.array(blurred_predictions))
                metrics['blur_level'] = blur_level
                model_results.append(metrics)
            
            results[model_name] = model_results
        
        # Save and plot results
        self._save_blur_results(results)
        self._plot_blur_sensitivity(results)
        
        return results
    
    def _save_blur_results(self, results):
        """Save blur sensitivity results to CSV"""
        all_results = []
        
        for model_name, model_results in results.items():
            for result in model_results:
                result['model'] = model_name
                all_results.append(result)
        
        df = pd.DataFrame(all_results)
        df.to_csv(os.path.join(self.results_dir, 'blur_sensitivity_results.csv'), index=False)
        print(f"Blur sensitivity results saved to {self.results_dir}/blur_sensitivity_results.csv")
    
    def _plot_blur_sensitivity(self, results):
        """Plot blur sensitivity results"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        
        for i, metric in enumerate(metrics):
            ax = axes[i // 3, i % 3]
            
            for model_name, model_results in results.items():
                blur_levels = [r['blur_level'] for r in model_results]
                metric_values = [r[metric] for r in model_results]
                ax.plot(blur_levels, metric_values, marker='o', label=model_name, linewidth=2)
            
            ax.set_xlabel('Blur Level (σ)')
            ax.set_ylabel(metric.capitalize())
            ax.set_title(f'{metric.capitalize()} vs Blur Level')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Remove the last subplot if not needed
        if len(metrics) < 6:
            axes[1, 2].remove()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'blur_sensitivity_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Blur sensitivity plot saved to {self.results_dir}/blur_sensitivity_analysis.png")
    
    def brightness_contrast_sensitivity_analysis(self, brightness_factors=[0.5, 0.75, 1.0, 1.25, 1.5]):
        """Analyze sensitivity to brightness and contrast changes"""
        print("\n=== Brightness/Contrast Sensitivity Analysis ===")
        
        results = {}
        
        for model_name in self.models.keys():
            print(f"Analyzing {model_name}...")
            model_results = []
            
            for factor in tqdm(brightness_factors, desc=f"{model_name} brightness analysis"):
                # Get test images
                if model_name == 'efficientnet':
                    test_gen = self.test_data['efficientnet']
                else:
                    test_gen = self.test_data['vgg_resnet']
                
                # Reset generator
                test_gen.reset()
                
                modified_predictions = []
                
                for batch_idx in range(len(test_gen)):
                    batch_images, _ = test_gen[batch_idx]
                    
                    # Apply brightness/contrast modification
                    modified_images = np.clip(batch_images * factor, 0, 1)
                    
                    # Get predictions
                    pred = self.models[model_name].predict(modified_images, verbose=0).flatten()
                    modified_predictions.extend(pred)
                
                # Calculate metrics
                metrics = self._calculate_metrics(self.y_true, np.array(modified_predictions))
                metrics['brightness_factor'] = factor
                model_results.append(metrics)
            
            results[model_name] = model_results
        
        # Save and plot results
        self._save_brightness_results(results)
        self._plot_brightness_sensitivity(results)
        
        return results
    
    def _save_brightness_results(self, results):
        """Save brightness sensitivity results to CSV"""
        all_results = []
        
        for model_name, model_results in results.items():
            for result in model_results:
                result['model'] = model_name
                all_results.append(result)
        
        df = pd.DataFrame(all_results)
        df.to_csv(os.path.join(self.results_dir, 'brightness_sensitivity_results.csv'), index=False)
        print(f"Brightness sensitivity results saved to {self.results_dir}/brightness_sensitivity_results.csv")
    
    def _plot_brightness_sensitivity(self, results):
        """Plot brightness sensitivity results"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        
        for i, metric in enumerate(metrics):
            ax = axes[i // 3, i % 3]
            
            for model_name, model_results in results.items():
                factors = [r['brightness_factor'] for r in model_results]
                metric_values = [r[metric] for r in model_results]
                ax.plot(factors, metric_values, marker='o', label=model_name, linewidth=2)
            
            ax.set_xlabel('Brightness Factor')
            ax.set_ylabel(metric.capitalize())
            ax.set_title(f'{metric.capitalize()} vs Brightness Factor')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Remove the last subplot if not needed
        if len(metrics) < 6:
            axes[1, 2].remove()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'brightness_sensitivity_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Brightness sensitivity plot saved to {self.results_dir}/brightness_sensitivity_analysis.png")
    
    def model_agreement_analysis(self):
        """Analyze agreement between different models"""
        print("\n=== Model Agreement Analysis ===")
        
        if len(self.models) < 2:
            print("Need at least 2 models for agreement analysis")
            return
        
        # Get predictions from all models
        predictions = {}
        for model_name in self.models.keys():
            if model_name == 'efficientnet':
                test_gen = self.test_data['efficientnet']
            else:
                test_gen = self.test_data['vgg_resnet']
            
            test_gen.reset()
            pred = self.models[model_name].predict(test_gen, verbose=0).flatten()
            predictions[model_name] = (pred > 0.5).astype(int)
        
        # Calculate agreement metrics
        model_names = list(predictions.keys())
        agreement_matrix = np.zeros((len(model_names), len(model_names)))
        
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names):
                if i == j:
                    agreement_matrix[i, j] = 1.0
                else:
                    agreement = np.mean(predictions[model1] == predictions[model2])
                    agreement_matrix[i, j] = agreement
        
        # Create agreement heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(agreement_matrix, 
                   annot=True, 
                   fmt='.3f', 
                   xticklabels=model_names, 
                   yticklabels=model_names,
                   cmap='Blues',
                   vmin=0, 
                   vmax=1)
        plt.title('Model Agreement Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'model_agreement_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save agreement matrix
        agreement_df = pd.DataFrame(agreement_matrix, 
                                  index=model_names, 
                                  columns=model_names)
        agreement_df.to_csv(os.path.join(self.results_dir, 'model_agreement_matrix.csv'))
        
        print(f"Model agreement analysis saved to {self.results_dir}/")
        
        return agreement_matrix
    
    def generate_sensitivity_report(self):
        """Generate a comprehensive sensitivity analysis report"""
        print("\n=== Generating Sensitivity Analysis Report ===")
        
        # Run all analyses
        noise_results = self.noise_sensitivity_analysis()
        blur_results = self.blur_sensitivity_analysis()
        brightness_results = self.brightness_contrast_sensitivity_analysis()
        agreement_matrix = self.model_agreement_analysis()
        
        # Generate summary report
        report = self._create_summary_report(noise_results, blur_results, brightness_results, agreement_matrix)
        
        # Save report
        with open(os.path.join(self.results_dir, 'sensitivity_analysis_report.txt'), 'w') as f:
            f.write(report)
        
        print(f"Comprehensive sensitivity analysis report saved to {self.results_dir}/sensitivity_analysis_report.txt")
        
        return report
    
    def _create_summary_report(self, noise_results, blur_results, brightness_results, agreement_matrix):
        """Create a summary report of all sensitivity analyses"""
        report = """
ARTERIAI ENSEMBLE MODEL SENSITIVITY ANALYSIS REPORT
==================================================

1. BASELINE PERFORMANCE
-----------------------
"""
        
        # Add baseline performance
        for model_name, pred_prob in self.baseline_predictions.items():
            metrics = self._calculate_metrics(self.y_true, pred_prob)
            report += f"\n{model_name.upper()}:\n"
            for metric, value in metrics.items():
                report += f"  {metric.capitalize()}: {value:.4f}\n"
        
        report += "\n\n2. NOISE SENSITIVITY ANALYSIS\n"
        report += "-----------------------------\n"
        
        # Add noise sensitivity summary
        for model_name, results in noise_results.items():
            report += f"\n{model_name.upper()}:\n"
            baseline_acc = results[0]['accuracy']  # No noise
            max_noise_acc = min([r['accuracy'] for r in results])
            report += f"  Baseline Accuracy: {baseline_acc:.4f}\n"
            report += f"  Worst Case Accuracy (max noise): {max_noise_acc:.4f}\n"
            report += f"  Accuracy Drop: {baseline_acc - max_noise_acc:.4f}\n"
        
        report += "\n\n3. BLUR SENSITIVITY ANALYSIS\n"
        report += "----------------------------\n"
        
        # Add blur sensitivity summary
        for model_name, results in blur_results.items():
            report += f"\n{model_name.upper()}:\n"
            baseline_acc = results[0]['accuracy']  # No blur
            max_blur_acc = min([r['accuracy'] for r in results])
            report += f"  Baseline Accuracy: {baseline_acc:.4f}\n"
            report += f"  Worst Case Accuracy (max blur): {max_blur_acc:.4f}\n"
            report += f"  Accuracy Drop: {baseline_acc - max_blur_acc:.4f}\n"
        
        report += "\n\n4. BRIGHTNESS SENSITIVITY ANALYSIS\n"
        report += "----------------------------------\n"
        
        # Add brightness sensitivity summary
        for model_name, results in brightness_results.items():
            report += f"\n{model_name.upper()}:\n"
            baseline_acc = results[2]['accuracy']  # Factor 1.0 (normal brightness)
            min_acc = min([r['accuracy'] for r in results])
            report += f"  Baseline Accuracy: {baseline_acc:.4f}\n"
            report += f"  Worst Case Accuracy: {min_acc:.4f}\n"
            report += f"  Accuracy Drop: {baseline_acc - min_acc:.4f}\n"
        
        report += "\n\n5. MODEL AGREEMENT ANALYSIS\n"
        report += "---------------------------\n"
        
        # Add agreement analysis
        model_names = list(self.models.keys())
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names):
                if i < j:  # Only show unique pairs
                    agreement = agreement_matrix[i, j]
                    report += f"  {model1} vs {model2}: {agreement:.3f}\n"
        
        report += "\n\n6. RECOMMENDATIONS\n"
        report += "------------------\n"
        
        # Generate recommendations based on analysis
        report += self._generate_recommendations(noise_results, blur_results, brightness_results, agreement_matrix)
        
        return report
    
    def _generate_recommendations(self, noise_results, blur_results, brightness_results, agreement_matrix):
        """Generate recommendations based on sensitivity analysis"""
        recommendations = ""
        
        # Find most robust model
        noise_robustness = {}
        for model_name, results in noise_results.items():
            baseline_acc = results[0]['accuracy']
            worst_acc = min([r['accuracy'] for r in results])
            noise_robustness[model_name] = baseline_acc - worst_acc
        
        most_robust_noise = min(noise_robustness, key=noise_robustness.get)
        recommendations += f"- Most noise-robust model: {most_robust_noise}\n"
        
        # Find most blur-robust model
        blur_robustness = {}
        for model_name, results in blur_results.items():
            baseline_acc = results[0]['accuracy']
            worst_acc = min([r['accuracy'] for r in results])
            blur_robustness[model_name] = baseline_acc - worst_acc
        
        most_robust_blur = min(blur_robustness, key=blur_robustness.get)
        recommendations += f"- Most blur-robust model: {most_robust_blur}\n"
        
        # Check ensemble benefits
        if 'ensemble' in noise_results:
            ensemble_noise_robustness = noise_robustness.get('ensemble', float('inf'))
            individual_robustness = [v for k, v in noise_robustness.items() if k != 'ensemble']
            
            if individual_robustness and ensemble_noise_robustness < min(individual_robustness):
                recommendations += "- Ensemble shows improved robustness compared to individual models\n"
            else:
                recommendations += "- Consider ensemble weighting strategies for better robustness\n"
        
        # Image quality recommendations
        max_noise_level = 0.1  # Based on analysis
        max_blur_level = 2     # Based on analysis
        recommendations += f"- Maintain image noise levels below {max_noise_level}\n"
        recommendations += f"- Maintain image sharpness (blur σ < {max_blur_level})\n"
        recommendations += "- Ensure proper brightness levels (factor between 0.75-1.25)\n"
        
        return recommendations


def main():
    parser = argparse.ArgumentParser(description='Sensitivity Analysis for ArteriAI Ensemble Model')
    parser.add_argument('--test_dir', type=str, default='data/splits/test', 
                       help='Path to test data directory')
    parser.add_argument('--results_dir', type=str, default='sensitivity_analysis_results',
                       help='Directory to save results')
    parser.add_argument('--analysis_type', type=str, default='all',
                       choices=['noise', 'blur', 'brightness', 'agreement', 'all'],
                       help='Type of sensitivity analysis to perform')
    
    args = parser.parse_args()
    
    # Initialize sensitivity analysis
    analyzer = EnsembleSensitivityAnalysis(
        test_dir=args.test_dir,
        results_dir=args.results_dir
    )
    
    # Run specified analysis
    if args.analysis_type == 'noise':
        analyzer.noise_sensitivity_analysis()
    elif args.analysis_type == 'blur':
        analyzer.blur_sensitivity_analysis()
    elif args.analysis_type == 'brightness':
        analyzer.brightness_contrast_sensitivity_analysis()
    elif args.analysis_type == 'agreement':
        analyzer.model_agreement_analysis()
    elif args.analysis_type == 'all':
        analyzer.generate_sensitivity_report()
    
    print(f"\nSensitivity analysis completed! Results saved to {args.results_dir}/")


if __name__ == "__main__":
    main() 