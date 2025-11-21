import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess_input

from evaluation_utils import compute_metrics, evaluate_model, print_evaluation_summary


MODEL_CONFIGS = [
    ('EfficientNetB0', 'models/efficientnet_final_model.keras', efficientnet_preprocess_input, None),
    ('ResNet50', 'models/resnet50_final_model.keras', None, 1.0 / 255.0),
    ('VGG16', 'models/vgg16_final_model.keras', None, 1.0 / 255.0),
]


if __name__ == "__main__":
    base_results = []
    for label, path, preprocess_fn, rescale in MODEL_CONFIGS:
        result = evaluate_model(
            model_label=label,
            model_path=path,
            preprocess_function=preprocess_fn,
            rescale=rescale,
            display=False,
        )
        print_evaluation_summary(result)
        base_results.append(result)

    y_true = base_results[0]['y_true']
    prediction_matrix = np.vstack([res['y_pred'] for res in base_results])
    probability_matrix = np.vstack([res['y_prob'] for res in base_results])

    votes = prediction_matrix.sum(axis=0)
    ensemble_pred = (votes >= 2).astype(int)  # Hard majority vote
    ensemble_prob = probability_matrix.mean(axis=0)  # Average probability for reporting/AUC

    ensemble_metrics = compute_metrics(y_true, ensemble_pred, ensemble_prob)
    ensemble_report = classification_report(y_true, ensemble_pred, target_names=['Benign', 'Malignant'], digits=4)
    ensemble_confusion = confusion_matrix(y_true, ensemble_pred)

    ensemble_result = {
        'label': 'Hard Voting Ensemble',
        'metrics': ensemble_metrics,
        'report': ensemble_report,
        'confusion_matrix': ensemble_confusion,
        'y_true': y_true,
        'y_pred': ensemble_pred,
        'y_prob': ensemble_prob,
    }

    print_evaluation_summary(ensemble_result)

