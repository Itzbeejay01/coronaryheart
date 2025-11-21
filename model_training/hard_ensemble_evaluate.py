import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)

# --- Argument parsing ---
def get_args():
    parser = argparse.ArgumentParser(description='Evaluate ensemble of VGG16, ResNet50, EfficientNet models.')
    parser.add_argument('--test_dir', type=str, default='data/splits/test', help='Path to test data directory')
    parser.add_argument('--results_dir', type=str, default='ensemble_results', help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--img_size', type=int, nargs=2, default=[224, 224], help='Image size (height width)')
    return parser.parse_args()

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

def main():
    args = get_args()

    base_dir = Path(__file__).resolve().parent.parent
    test_dir = Path(args.test_dir)
    if not test_dir.is_absolute():
        test_dir = base_dir / test_dir

    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = base_dir / results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    batch_size = args.batch_size
    img_size = tuple(args.img_size)

    print(f"Test data directory: {test_dir}")
    print(f"Results will be saved to: {results_dir}")

    vgg_model_path = base_dir / "models" / "vgg16_final_model.keras"
    resnet_model_path = base_dir / "models" / "resnet50_final_model.keras"
    eff_model_path = base_dir / "models" / "efficientnet_final_model.keras"

    print(f"Loading VGG16 from: {vgg_model_path}")
    print(f"Loading ResNet50 from: {resnet_model_path}")
    print(f"Loading EfficientNet from: {eff_model_path}")

    vgg16_model = keras.models.load_model(vgg_model_path, compile=False)
    resnet50_model = keras.models.load_model(resnet_model_path, compile=False)
    efficientnet_model = keras.models.load_model(eff_model_path, compile=False)

    print(f"✓ VGG16 loaded: {vgg16_model.count_params():,} parameters")
    print(f"✓ ResNet50 loaded: {resnet50_model.count_params():,} parameters")
    print(f"✓ EfficientNet loaded: {efficientnet_model.count_params():,} parameters")

    rescale_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_gen_vgg_resnet = rescale_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False,
    )

    efficientnet_datagen = ImageDataGenerator(preprocessing_function=efficientnet_preprocess)
    test_gen_efficientnet = efficientnet_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False,
    )

    y_true = test_gen_vgg_resnet.classes
    class_names = list(test_gen_vgg_resnet.class_indices.keys())

    print("Predicting with VGG16...")
    vgg16_prob = vgg16_model.predict(test_gen_vgg_resnet, verbose=1).flatten()
    print("Predicting with ResNet50...")
    resnet_prob = resnet50_model.predict(test_gen_vgg_resnet, verbose=1).flatten()
    print("Predicting with EfficientNet...")
    efficientnet_prob = efficientnet_model.predict(test_gen_efficientnet, verbose=1).flatten()

    vgg16_pred = (vgg16_prob > 0.5).astype(int)
    resnet_pred = (resnet_prob > 0.5).astype(int)
    efficientnet_pred = (efficientnet_prob > 0.5).astype(int)

    votes = np.vstack([vgg16_pred, resnet_pred, efficientnet_pred])
    ensemble_pred = (np.sum(votes, axis=0) >= 2).astype(int)
    ensemble_prob = np.mean([vgg16_prob, resnet_prob, efficientnet_prob], axis=0)

    results = []

    vgg16_metrics = get_metrics(y_true, vgg16_pred, vgg16_prob)
    print_and_save_metrics("VGG16", vgg16_metrics, results)
    plot_roc_curve(y_true, vgg16_prob, "VGG16", color="green")

    resnet_metrics = get_metrics(y_true, resnet_pred, resnet_prob)
    print_and_save_metrics("ResNet50", resnet_metrics, results)
    plot_roc_curve(y_true, resnet_prob, "ResNet50", color="red")

    efficientnet_metrics = get_metrics(y_true, efficientnet_pred, efficientnet_prob)
    print_and_save_metrics("EfficientNet", efficientnet_metrics, results)
    plot_roc_curve(y_true, efficientnet_prob, "EfficientNet", color="gold")

    ensemble_metrics = get_metrics(y_true, ensemble_pred, ensemble_prob)
    print_and_save_metrics("Ensemble", ensemble_metrics, results)
    plot_roc_curve(y_true, ensemble_prob, "Ensemble", color="blue")

    plot_ensemble_roc_with_bases(
        y_true,
        vgg16_prob,
        resnet_prob,
        efficientnet_prob,
        ensemble_prob,
    )

    results_df = pd.DataFrame(results)
    results_df.to_csv(results_dir / "ensemble_evaluation_metrics.csv", index=False)

    print(f"\nAll results, confusion matrices, and ROC curves saved in: {results_dir}")


if __name__ == "__main__":
    main()