import os
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def create_generator(
    preprocess_function: Optional[Callable],
    test_dir: str,
    img_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    rescale: Optional[float] = None,
):
    """Create a deterministic test generator for a given preprocessing function."""
    datagen = ImageDataGenerator(
        preprocessing_function=preprocess_function,
        rescale=rescale,
    )
    generator = datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False,
    )
    return generator


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Compute standard binary classification metrics with safe fallbacks."""
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall_pos = recall_score(y_true, y_pred, zero_division=0)
    recall_neg = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'sensitivity': recall_pos,
        'specificity': recall_neg,
        'f1': f1,
        'auc': auc,
    }


def print_evaluation_summary(result: Dict) -> None:
    """Pretty-print evaluation metrics, classification report, and confusion matrix."""
    label = result['label']
    metrics = result['metrics']
    print(f"\n===== {label} Performance Metrics =====")
    print(f"Accuracy   : {metrics['accuracy']:.4f}")
    print(f"Precision  : {metrics['precision']:.4f}")
    print(f"Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(f"F1-Score   : {metrics['f1']:.4f}")
    print(f"AUC-ROC    : {metrics['auc']:.4f}")

    print("\nClassification Report:")
    print(result['report'])

    print("\nConfusion Matrix:")
    print(result['confusion_matrix'])


def evaluate_model(
    model_label: str,
    model_path: str,
    preprocess_function: Callable,
    test_dir: str = 'data/splits/test',
    img_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    weights_path: Optional[str] = None,
    rescale: Optional[float] = None,
    display: bool = True,
) -> Dict:
    """
    Evaluate a binary classifier on the test split and return detailed results.

    Returns a dictionary containing metrics, predictions, and artefacts required
    for further downstream analysis.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    generator = create_generator(preprocess_function, test_dir, img_size, batch_size, rescale=rescale)
    model = tf.keras.models.load_model(model_path)

    if weights_path and os.path.exists(weights_path):
        model.load_weights(weights_path)

    y_true = generator.classes
    generator.reset()
    y_prob = model.predict(generator, verbose=1).flatten()
    y_pred = (y_prob > 0.5).astype(int)

    metrics = compute_metrics(y_true, y_pred, y_prob)
    report = classification_report(y_true, y_pred, target_names=['Benign', 'Malignant'], digits=4)
    matrix = confusion_matrix(y_true, y_pred)

    result = {
        'label': model_label,
        'metrics': metrics,
        'report': report,
        'confusion_matrix': matrix,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_prob': y_prob,
    }

    if display:
        print_evaluation_summary(result)

    return result

