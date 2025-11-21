import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from tensorflow.keras.models import load_model

# --- Argument parsing ---
def get_args():
    parser = argparse.ArgumentParser(description='Evaluate ensemble of VGG16, ResNet50, EfficientNet models.')
    parser.add_argument('--test_dir', type=str, default='data/splits/test', help='Path to test data directory')
    parser.add_argument('--results_dir', type=str, default='ensemble_results', help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--img_size', type=int, nargs=2, default=[224, 224], help='Image size (height width)')
    return parser.parse_args()

args = get_args()

TEST_DIR = args.test_dir
RESULTS_DIR = args.results_dir
BATCH_SIZE = args.batch_size
IMG_SIZE = tuple(args.img_size)
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"Test data directory: {TEST_DIR}")
print(f"Results will be saved to: {RESULTS_DIR}")

# --- Load all three final trained models ---
VGG16_MODEL_PATH = 'models/vgg16_final_model.keras'
RESNET50_MODEL_PATH = 'models/resnet50_final_model.keras'
EFFICIENTNET_MODEL_PATH = 'models/efficientnet_final_model.keras'

print(f"Loading VGG16 from: {VGG16_MODEL_PATH}")
print(f"Loading ResNet50 from: {RESNET50_MODEL_PATH}")
print(f"Loading EfficientNet from: {EFFICIENTNET_MODEL_PATH}")

# --- Load all models from their final .keras files ---
vgg16_model = load_model(VGG16_MODEL_PATH, compile=False)
resnet50_model = load_model(RESNET50_MODEL_PATH, compile=False)
efficientnet_model = load_model(EFFICIENTNET_MODEL_PATH, compile=False)

print(f"✓ VGG16 loaded: {vgg16_model.count_params():,} parameters")
print(f"✓ ResNet50 loaded: {resnet50_model.count_params():,} parameters")
print(f"✓ EfficientNet loaded: {efficientnet_model.count_params():,} parameters")

# --- Data Generators ---
val_test_datagen_vgg_resnet = ImageDataGenerator(rescale=1./255)
test_gen_vgg_resnet = val_test_datagen_vgg_resnet.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)
val_test_datagen_efficientnet = ImageDataGenerator(preprocessing_function=efficientnet_preprocess)
test_gen_efficientnet = val_test_datagen_efficientnet.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# --- Get true labels ---
y_true = test_gen_vgg_resnet.classes
class_names = list(test_gen_vgg_resnet.class_indices.keys())

# --- Get predictions ---
print('Predicting with VGG16...')
vgg16_pred_prob = vgg16_model.predict(test_gen_vgg_resnet, verbose=1).flatten()
print('Predicting with ResNet50...')
resnet50_pred_prob = resnet50_model.predict(test_gen_vgg_resnet, verbose=1).flatten()
print('Predicting with EfficientNet...')
efficientnet_pred_prob = efficientnet_model.predict(test_gen_efficientnet, verbose=1).flatten()

# --- Ensemble prediction (average probabilities) ---
ensemble_pred_prob = (vgg16_pred_prob + resnet50_pred_prob + efficientnet_pred_prob) / 3.0

# --- Binarize predictions ---
vgg16_pred = (vgg16_pred_prob > 0.5).astype(int)
resnet50_pred = (resnet50_pred_prob > 0.5).astype(int)
efficientnet_pred = (efficientnet_pred_prob > 0.5).astype(int)
ensemble_pred = (ensemble_pred_prob > 0.5).astype(int)

# --- Evaluation function ---
def get_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    sens = recall_score(y_true, y_pred)  # Sensitivity (Recall)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate specificity (True Negative Rate)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        'accuracy': acc,
        'precision': prec,
        'sensitivity': sens,
        'specificity': specificity,
        'f1': f1,
        'roc_auc': auc,
        'confusion_matrix': cm
    }

def print_and_save_metrics(name, metrics, results_list):
    print(f'\n{name} Results:')
    print(f'  Accuracy   : {metrics["accuracy"]:.4f}')
    print(f'  Precision  : {metrics["precision"]:.4f}')
    print(f'  Sensitivity: {metrics["sensitivity"]:.4f}')
    print(f'  Specificity: {metrics["specificity"]:.4f}')
    print(f'  F1-Score   : {metrics["f1"]:.4f}')
    print(f'  ROC-AUC    : {metrics["roc_auc"]:.4f}')
    print(f'  Confusion Matrix:\n{metrics["confusion_matrix"]}')
    # Save metrics to results list for CSV
    results_list.append({
        'model': name,
        'accuracy': metrics['accuracy'],
        'precision': metrics['precision'],
        'sensitivity': metrics['sensitivity'],
        'specificity': metrics['specificity'],
        'f1': metrics['f1'],
        'roc_auc': metrics['roc_auc']
    })
    # Save confusion matrix as image
    plot_confusion_matrix(metrics['confusion_matrix'], class_names, name)

# --- Plotting functions ---
def plot_confusion_matrix(cm, class_names, model_name):
    plt.figure(figsize=(4,4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'{model_name} Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'{model_name}_confusion_matrix.png'))
    plt.close()

def plot_roc_curve(y_true, y_prob, model_name, color=None):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.4f})', color=color)
    return fpr, tpr, auc

def plot_ensemble_roc_with_bases(y_true, vgg16_prob, resnet50_prob, efficientnet_prob, ensemble_prob):
    plt.figure()
    # Plot each base model
    plot_roc_curve(y_true, vgg16_prob, 'VGG16', color='green')
    plot_roc_curve(y_true, resnet50_prob, 'ResNet50', color='red')
    plot_roc_curve(y_true, efficientnet_prob, 'EfficientNet', color='gold')
    # Plot ensemble
    plot_roc_curve(y_true, ensemble_prob, 'Ensemble', color='blue')
    plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve: All Models')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(os.path.join(RESULTS_DIR, 'ensemble_and_base_roc_curve.png'))
    plt.close()

# --- Collect and save all results ---
results = []

# VGG16
vgg16_metrics = get_metrics(y_true, vgg16_pred, vgg16_pred_prob)
print_and_save_metrics('VGG16', vgg16_metrics, results)
plot_roc_curve(y_true, vgg16_pred_prob, 'VGG16', color='green')

# ResNet50
resnet50_metrics = get_metrics(y_true, resnet50_pred, resnet50_pred_prob)
print_and_save_metrics('ResNet50', resnet50_metrics, results)
plot_roc_curve(y_true, resnet50_pred_prob, 'ResNet50', color='red')

# EfficientNet
efficientnet_metrics = get_metrics(y_true, efficientnet_pred, efficientnet_pred_prob)
print_and_save_metrics('EfficientNet', efficientnet_metrics, results)
plot_roc_curve(y_true, efficientnet_pred_prob, 'EfficientNet', color='gold')

# Ensemble
ensemble_metrics = get_metrics(y_true, ensemble_pred, ensemble_pred_prob)
print_and_save_metrics('Ensemble', ensemble_metrics, results)
plot_roc_curve(y_true, ensemble_pred_prob, 'Ensemble', color='blue')

# Plot all ROC curves together for ensemble
plot_ensemble_roc_with_bases(y_true, vgg16_pred_prob, resnet50_pred_prob, efficientnet_pred_prob, ensemble_pred_prob)

# Save all metrics to CSV
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(RESULTS_DIR, 'ensemble_evaluation_metrics.csv'), index=False)

print(f'\nAll results, confusion matrices, and ROC curves saved in: {RESULTS_DIR}') 