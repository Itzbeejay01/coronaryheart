import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def evaluate_classification(y_true, y_pred, y_prob=None):
    results = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred)
    }
    if y_prob is not None:
        results['roc_auc'] = roc_auc_score(y_true, y_prob)
    return results

def print_evaluation(results):
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

def sensitivity_analysis(model, X, y, param_grid):
    # Placeholder for hyperparameter sensitivity analysis
    print("Sensitivity analysis not implemented. Provide param_grid and implement grid search.")

def perturbation_test(model, X, y, noise_level=0.1):
    # Placeholder for perturbation test
    X_noisy = X + noise_level * np.random.normal(size=X.shape)
    y_pred = model.predict(X_noisy)
    print("Perturbation test results:")
    print_evaluation(evaluate_classification(y, y_pred))

# Example usage (uncomment and adapt as needed):
# results = evaluate_classification(y_true, y_pred, y_prob)
# print_evaluation(results)
# sensitivity_analysis(model, X, y, param_grid)
# perturbation_test(model, X, y) 