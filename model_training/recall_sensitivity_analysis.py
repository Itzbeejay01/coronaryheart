import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import cv2
from scipy import ndimage
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class RecallSensitivityAnalysis:
    """
    Recall (Sensitivity) Analysis Class
    
    This class specifically analyzes how the recall (sensitivity) of models changes
    under different conditions and perturbations. Recall is crucial in medical
    applications as it measures the ability to correctly identify positive cases.
    
    Recall = TP / (TP + FN) = True Positives / All Actual Positives
    """
    
    def __init__(self, test_dir='data/splits/test', results_dir='recall_sensitivity_results'):
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
        
        # Baseline predictions and recall
        self.baseline_predictions = self._get_baseline_predictions()
        self.baseline_recalls = self._calculate_baseline_recalls()
        
        print(f"Baseline recalls: {self.baseline_recalls}")
        
    def _load_models(self):
        """Load the final trained models from models folder"""
        print("Loading final trained models...")
        
        models = {}
        
        # Load final trained models
        model_paths = {
            'vgg16': 'models/vgg16_final_model.keras',
            'resnet50': 'models/resnet50_final_model.keras',
            'efficientnet': 'models/efficientnet_final_model.keras'
        }
        
        for model_name, model_path in model_paths.items():
            if os.path.exists(model_path):
                try:
                    models[model_name] = load_model(model_path, compile=False)
                    print(f"✓ {model_name.upper()} loaded from {model_path}")
                except Exception as e:
                    print(f"✗ Failed to load {model_name} from {model_path}: {e}")
            else:
                print(f"⚠ {model_name.upper()} model not found at {model_path}")
        
        if not models:
            print("❌ No models could be loaded.")
        else:
            print(f"\n✓ Successfully loaded {len(models)} models for recall analysis")
        
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
    
    def _calculate_baseline_recalls(self):
        """Calculate baseline recall for each model"""
        recalls = {}
        
        for model_name, pred_prob in self.baseline_predictions.items():
            y_pred = (pred_prob > 0.5).astype(int)
            recall = recall_score(self.y_true, y_pred)
            recalls[model_name] = recall
        
        return recalls
    
    def _calculate_recall_metrics(self, y_true, y_pred_prob, threshold=0.5):
        """Calculate recall and related metrics"""
        y_pred = (y_pred_prob > threshold).astype(int)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            # Handle edge cases
            if len(cm) == 1:
                if y_true[0] == 0:  # All negatives
                    tn, fp, fn, tp = len(y_true), 0, 0, 0
                else:  # All positives
                    tn, fp, fn, tp = 0, 0, 0, len(y_true)
            else:
                tn, fp, fn, tp = 0, 0, 0, 0
        
        # Calculate metrics
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'recall': recall,
            'precision': precision,
            'specificity': specificity,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn,
            'total_positives': tp + fn,
            'total_negatives': tn + fp
        }
    
    def threshold_recall_analysis(self, thresholds=np.arange(0.1, 1.0, 0.05)):
        """
        Analyze how recall changes with different classification thresholds
        
        This is crucial for medical applications where we need to balance
        between missing positive cases (false negatives) and false alarms.
        """
        print("\n=== Threshold-Recall Analysis ===")
        print("Analyzing how recall changes with different classification thresholds...")
        
        results = {}
        
        for model_name in self.models.keys():
            print(f"Analyzing {model_name}...")
            model_results = []
            
            pred_prob = self.baseline_predictions[model_name]
            
            for threshold in tqdm(thresholds, desc=f"{model_name} threshold analysis"):
                metrics = self._calculate_recall_metrics(self.y_true, pred_prob, threshold)
                metrics['threshold'] = threshold
                model_results.append(metrics)
            
            results[model_name] = model_results
        
        # Ensemble analysis
        if 'ensemble' in self.baseline_predictions:
            print("Analyzing ensemble...")
            ensemble_results = []
            pred_prob = self.baseline_predictions['ensemble']
            
            for threshold in tqdm(thresholds, desc="Ensemble threshold analysis"):
                metrics = self._calculate_recall_metrics(self.y_true, pred_prob, threshold)
                metrics['threshold'] = threshold
                ensemble_results.append(metrics)
            
            results['ensemble'] = ensemble_results
        
        # Save and plot results
        self._save_threshold_results(results)
        self._plot_threshold_recall_curves(results)
        
        return results
    
    def noise_recall_analysis(self, noise_levels=[0.01, 0.05, 0.1, 0.15, 0.2]):
        """
        Analyze how recall changes with different levels of Gaussian noise
        
        This helps understand model robustness to image quality degradation.
        """
        print("\n=== Noise-Recall Analysis ===")
        print("Analyzing how recall changes with different noise levels...")
        
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
                
                # Calculate recall metrics
                metrics = self._calculate_recall_metrics(self.y_true, np.array(noisy_predictions))
                metrics['noise_level'] = noise_level
                model_results.append(metrics)
            
            results[model_name] = model_results
        
        # Save and plot results
        self._save_noise_results(results)
        self._plot_noise_recall_curves(results)
        
        return results
    
    def blur_recall_analysis(self, blur_levels=[0, 1, 2, 3, 4, 5]):
        """
        Analyze how recall changes with different levels of Gaussian blur
        
        This helps understand model performance on images with different sharpness levels.
        """
        print("\n=== Blur-Recall Analysis ===")
        print("Analyzing how recall changes with different blur levels...")
        
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
                
                # Calculate recall metrics
                metrics = self._calculate_recall_metrics(self.y_true, np.array(blurred_predictions))
                metrics['blur_level'] = blur_level
                model_results.append(metrics)
            
            results[model_name] = model_results
        
        # Save and plot results
        self._save_blur_results(results)
        self._plot_blur_recall_curves(results)
        
        return results
    
    def brightness_recall_analysis(self, brightness_factors=[0.5, 0.75, 1.0, 1.25, 1.5]):
        """
        Analyze how recall changes with different brightness levels
        
        This helps understand model performance under different lighting conditions.
        """
        print("\n=== Brightness-Recall Analysis ===")
        print("Analyzing how recall changes with different brightness levels...")
        
        results = {}
        
        for model_name in self.models.keys():
            print(f"Analyzing {model_name}...")
            model_results = []
            
            for brightness_factor in tqdm(brightness_factors, desc=f"{model_name} brightness analysis"):
                # Get test images
                if model_name == 'efficientnet':
                    test_gen = self.test_data['efficientnet']
                else:
                    test_gen = self.test_data['vgg_resnet']
                
                # Reset generator
                test_gen.reset()
                
                brightness_predictions = []
                
                for batch_idx in range(len(test_gen)):
                    batch_images, _ = test_gen[batch_idx]
                    
                    # Adjust brightness
                    if brightness_factor != 1.0:
                        brightness_images = np.clip(batch_images * brightness_factor, 0, 1)
                    else:
                        brightness_images = batch_images
                    
                    # Get predictions
                    pred = self.models[model_name].predict(brightness_images, verbose=0).flatten()
                    brightness_predictions.extend(pred)
                
                # Calculate recall metrics
                metrics = self._calculate_recall_metrics(self.y_true, np.array(brightness_predictions))
                metrics['brightness_factor'] = brightness_factor
                model_results.append(metrics)
            
            results[model_name] = model_results
        
        # Save and plot results
        self._save_brightness_results(results)
        self._plot_brightness_recall_curves(results)
        
        return results
    
    def _save_threshold_results(self, results):
        """Save threshold analysis results"""
        all_results = []
        
        for model_name, model_results in results.items():
            for result in model_results:
                result['model'] = model_name
                all_results.append(result)
        
        df = pd.DataFrame(all_results)
        output_file = os.path.join(self.results_dir, 'threshold_recall_analysis.csv')
        df.to_csv(output_file, index=False)
        print(f"Threshold analysis results saved to {output_file}")
    
    def _save_noise_results(self, results):
        """Save noise analysis results"""
        all_results = []
        
        for model_name, model_results in results.items():
            for result in model_results:
                result['model'] = model_name
                all_results.append(result)
        
        df = pd.DataFrame(all_results)
        output_file = os.path.join(self.results_dir, 'noise_recall_analysis.csv')
        df.to_csv(output_file, index=False)
        print(f"Noise analysis results saved to {output_file}")
    
    def _save_blur_results(self, results):
        """Save blur analysis results"""
        all_results = []
        
        for model_name, model_results in results.items():
            for result in model_results:
                result['model'] = model_name
                all_results.append(result)
        
        df = pd.DataFrame(all_results)
        output_file = os.path.join(self.results_dir, 'blur_recall_analysis.csv')
        df.to_csv(output_file, index=False)
        print(f"Blur analysis results saved to {output_file}")
    
    def _save_brightness_results(self, results):
        """Save brightness analysis results"""
        all_results = []
        
        for model_name, model_results in results.items():
            for result in model_results:
                result['model'] = model_name
                all_results.append(result)
        
        df = pd.DataFrame(all_results)
        output_file = os.path.join(self.results_dir, 'brightness_recall_analysis.csv')
        df.to_csv(output_file, index=False)
        print(f"Brightness analysis results saved to {output_file}")
    
    def _plot_threshold_recall_curves(self, results):
        """Plot threshold-recall curves"""
        plt.figure(figsize=(12, 8))
        
        colors = {'vgg16': 'green', 'resnet50': 'red', 'efficientnet': 'gold', 'ensemble': 'blue'}
        
        for model_name, model_results in results.items():
            thresholds = [r['threshold'] for r in model_results]
            recalls = [r['recall'] for r in model_results]
            precisions = [r['precision'] for r in model_results]
            
            plt.subplot(2, 2, 1)
            plt.plot(thresholds, recalls, label=f'{model_name}', color=colors.get(model_name, 'black'), linewidth=2)
            plt.xlabel('Classification Threshold')
            plt.ylabel('Recall (Sensitivity)')
            plt.title('Threshold vs Recall')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(2, 2, 2)
            plt.plot(thresholds, precisions, label=f'{model_name}', color=colors.get(model_name, 'black'), linewidth=2)
            plt.xlabel('Classification Threshold')
            plt.ylabel('Precision')
            plt.title('Threshold vs Precision')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(2, 2, 3)
            plt.plot(recalls, precisions, label=f'{model_name}', color=colors.get(model_name, 'black'), linewidth=2)
            plt.xlabel('Recall (Sensitivity)')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(2, 2, 4)
            f1_scores = [r['f1'] for r in model_results]
            plt.plot(thresholds, f1_scores, label=f'{model_name}', color=colors.get(model_name, 'black'), linewidth=2)
            plt.xlabel('Classification Threshold')
            plt.ylabel('F1-Score')
            plt.title('Threshold vs F1-Score')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'threshold_recall_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Threshold-recall curves saved to {self.results_dir}/threshold_recall_analysis.png")
    
    def _plot_noise_recall_curves(self, results):
        """Plot noise-recall curves"""
        plt.figure(figsize=(12, 8))
        
        colors = {'vgg16': 'green', 'resnet50': 'red', 'efficientnet': 'gold', 'ensemble': 'blue'}
        
        for model_name, model_results in results.items():
            noise_levels = [r['noise_level'] for r in model_results]
            recalls = [r['recall'] for r in model_results]
            precisions = [r['precision'] for r in model_results]
            f1_scores = [r['f1'] for r in model_results]
            
            plt.subplot(2, 2, 1)
            plt.plot(noise_levels, recalls, 'o-', label=f'{model_name}', color=colors.get(model_name, 'black'), linewidth=2)
            plt.xlabel('Noise Level (σ)')
            plt.ylabel('Recall (Sensitivity)')
            plt.title('Noise vs Recall')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(2, 2, 2)
            plt.plot(noise_levels, precisions, 'o-', label=f'{model_name}', color=colors.get(model_name, 'black'), linewidth=2)
            plt.xlabel('Noise Level (σ)')
            plt.ylabel('Precision')
            plt.title('Noise vs Precision')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(2, 2, 3)
            plt.plot(noise_levels, f1_scores, 'o-', label=f'{model_name}', color=colors.get(model_name, 'black'), linewidth=2)
            plt.xlabel('Noise Level (σ)')
            plt.ylabel('F1-Score')
            plt.title('Noise vs F1-Score')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(2, 2, 4)
            # Plot recall degradation from baseline
            baseline_recall = self.baseline_recalls.get(model_name, 0)
            recall_degradation = [baseline_recall - r for r in recalls]
            plt.plot(noise_levels, recall_degradation, 'o-', label=f'{model_name}', color=colors.get(model_name, 'black'), linewidth=2)
            plt.xlabel('Noise Level (σ)')
            plt.ylabel('Recall Degradation')
            plt.title('Noise vs Recall Degradation')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'noise_recall_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Noise-recall curves saved to {self.results_dir}/noise_recall_analysis.png")
    
    def _plot_blur_recall_curves(self, results):
        """Plot blur-recall curves"""
        plt.figure(figsize=(12, 8))
        
        colors = {'vgg16': 'green', 'resnet50': 'red', 'efficientnet': 'gold', 'ensemble': 'blue'}
        
        for model_name, model_results in results.items():
            blur_levels = [r['blur_level'] for r in model_results]
            recalls = [r['recall'] for r in model_results]
            precisions = [r['precision'] for r in model_results]
            f1_scores = [r['f1'] for r in model_results]
            
            plt.subplot(2, 2, 1)
            plt.plot(blur_levels, recalls, 'o-', label=f'{model_name}', color=colors.get(model_name, 'black'), linewidth=2)
            plt.xlabel('Blur Level (σ)')
            plt.ylabel('Recall (Sensitivity)')
            plt.title('Blur vs Recall')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(2, 2, 2)
            plt.plot(blur_levels, precisions, 'o-', label=f'{model_name}', color=colors.get(model_name, 'black'), linewidth=2)
            plt.xlabel('Blur Level (σ)')
            plt.ylabel('Precision')
            plt.title('Blur vs Precision')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(2, 2, 3)
            plt.plot(blur_levels, f1_scores, 'o-', label=f'{model_name}', color=colors.get(model_name, 'black'), linewidth=2)
            plt.xlabel('Blur Level (σ)')
            plt.ylabel('F1-Score')
            plt.title('Blur vs F1-Score')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(2, 2, 4)
            # Plot recall degradation from baseline
            baseline_recall = self.baseline_recalls.get(model_name, 0)
            recall_degradation = [baseline_recall - r for r in recalls]
            plt.plot(blur_levels, recall_degradation, 'o-', label=f'{model_name}', color=colors.get(model_name, 'black'), linewidth=2)
            plt.xlabel('Blur Level (σ)')
            plt.ylabel('Recall Degradation')
            plt.title('Blur vs Recall Degradation')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'blur_recall_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Blur-recall curves saved to {self.results_dir}/blur_recall_analysis.png")
    
    def _plot_brightness_recall_curves(self, results):
        """Plot brightness-recall curves"""
        plt.figure(figsize=(12, 8))
        
        colors = {'vgg16': 'green', 'resnet50': 'red', 'efficientnet': 'gold', 'ensemble': 'blue'}
        
        for model_name, model_results in results.items():
            brightness_factors = [r['brightness_factor'] for r in model_results]
            recalls = [r['recall'] for r in model_results]
            precisions = [r['precision'] for r in model_results]
            f1_scores = [r['f1'] for r in model_results]
            
            plt.subplot(2, 2, 1)
            plt.plot(brightness_factors, recalls, 'o-', label=f'{model_name}', color=colors.get(model_name, 'black'), linewidth=2)
            plt.xlabel('Brightness Factor')
            plt.ylabel('Recall (Sensitivity)')
            plt.title('Brightness vs Recall')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(2, 2, 2)
            plt.plot(brightness_factors, precisions, 'o-', label=f'{model_name}', color=colors.get(model_name, 'black'), linewidth=2)
            plt.xlabel('Brightness Factor')
            plt.ylabel('Precision')
            plt.title('Brightness vs Precision')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(2, 2, 3)
            plt.plot(brightness_factors, f1_scores, 'o-', label=f'{model_name}', color=colors.get(model_name, 'black'), linewidth=2)
            plt.xlabel('Brightness Factor')
            plt.ylabel('F1-Score')
            plt.title('Brightness vs F1-Score')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(2, 2, 4)
            # Plot recall degradation from baseline
            baseline_recall = self.baseline_recalls.get(model_name, 0)
            recall_degradation = [baseline_recall - r for r in recalls]
            plt.plot(brightness_factors, recall_degradation, 'o-', label=f'{model_name}', color=colors.get(model_name, 'black'), linewidth=2)
            plt.xlabel('Brightness Factor')
            plt.ylabel('Recall Degradation')
            plt.title('Brightness vs Recall Degradation')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'brightness_recall_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Brightness-recall curves saved to {self.results_dir}/brightness_recall_analysis.png")
    
    def generate_recall_report(self):
        """Generate comprehensive recall analysis report"""
        print("\n=== Generating Recall Analysis Report ===")
        
        # Run all analyses
        threshold_results = self.threshold_recall_analysis()
        noise_results = self.noise_recall_analysis()
        blur_results = self.blur_recall_analysis()
        brightness_results = self.brightness_recall_analysis()
        
        # Create summary report
        self._create_recall_summary_report(threshold_results, noise_results, blur_results, brightness_results)
        
        print(f"\nRecall analysis completed! Results saved in: {self.results_dir}")
    
    def _create_recall_summary_report(self, threshold_results, noise_results, blur_results, brightness_results):
        """Create a comprehensive summary report"""
        report_file = os.path.join(self.results_dir, 'recall_analysis_summary.txt')
        
        with open(report_file, 'w') as f:
            f.write("=== RECALL (SENSITIVITY) ANALYSIS SUMMARY ===\n\n")
            f.write("This report analyzes how the recall (sensitivity) of models changes under different conditions.\n")
            f.write("Recall is crucial in medical applications as it measures the ability to correctly identify positive cases.\n")
            f.write("Recall = TP / (TP + FN) = True Positives / All Actual Positives\n\n")
            
            # Baseline recalls
            f.write("BASELINE RECALLS:\n")
            f.write("-" * 50 + "\n")
            for model_name, recall in self.baseline_recalls.items():
                f.write(f"{model_name.upper()}: {recall:.4f}\n")
            f.write("\n")
            
            # Threshold analysis summary
            f.write("THRESHOLD ANALYSIS SUMMARY:\n")
            f.write("-" * 50 + "\n")
            f.write("This analysis shows how recall changes with different classification thresholds.\n")
            f.write("Lower thresholds generally increase recall but may decrease precision.\n\n")
            
            for model_name, model_results in threshold_results.items():
                max_recall_idx = np.argmax([r['recall'] for r in model_results])
                max_recall_result = model_results[max_recall_idx]
                f.write(f"{model_name.upper()}:\n")
                f.write(f"  Max Recall: {max_recall_result['recall']:.4f} at threshold {max_recall_result['threshold']:.2f}\n")
                f.write(f"  Corresponding Precision: {max_recall_result['precision']:.4f}\n")
                f.write(f"  Corresponding F1-Score: {max_recall_result['f1']:.4f}\n\n")
            
            # Noise analysis summary
            f.write("NOISE ANALYSIS SUMMARY:\n")
            f.write("-" * 50 + "\n")
            f.write("This analysis shows how recall degrades with increasing noise levels.\n\n")
            
            for model_name, model_results in noise_results.items():
                baseline_recall = self.baseline_recalls.get(model_name, 0)
                worst_recall = min([r['recall'] for r in model_results])
                worst_noise = [r['noise_level'] for r in model_results if r['recall'] == worst_recall][0]
                f.write(f"{model_name.upper()}:\n")
                f.write(f"  Baseline Recall: {baseline_recall:.4f}\n")
                f.write(f"  Worst Recall: {worst_recall:.4f} at noise level {worst_noise}\n")
                f.write(f"  Recall Degradation: {baseline_recall - worst_recall:.4f}\n\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS:\n")
            f.write("-" * 50 + "\n")
            f.write("1. For medical applications, prioritize models with higher baseline recall.\n")
            f.write("2. Consider using different thresholds for different clinical scenarios.\n")
            f.write("3. Monitor image quality to maintain optimal recall performance.\n")
            f.write("4. Use ensemble methods to improve recall robustness.\n")
            f.write("5. Regular model retraining with diverse data can help maintain recall performance.\n")
        
        print(f"Recall analysis summary saved to {report_file}")

def main():
    parser = argparse.ArgumentParser(description='Analyze recall (sensitivity) of models under different conditions')
    parser.add_argument('--test_dir', type=str, default='data/splits/test', help='Path to test data directory')
    parser.add_argument('--results_dir', type=str, default='recall_sensitivity_results', help='Directory to save results')
    
    args = parser.parse_args()
    
    # Initialize recall analysis
    analyzer = RecallSensitivityAnalysis(args.test_dir, args.results_dir)
    
    # Generate comprehensive report
    analyzer.generate_recall_report()

if __name__ == "__main__":
    main() 