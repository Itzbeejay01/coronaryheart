from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess_input

from evaluation_utils import evaluate_model, print_evaluation_summary


model_path = 'models/efficientnet_final_model.keras'  # Full trained model
weights_path = 'model_training/checkpoints/efficientnet_best_model.weights.h5'  # Optional best weights

if __name__ == "__main__":
    result = evaluate_model(
        model_label='EfficientNetB0',
        model_path=model_path,
        preprocess_function=efficientnet_preprocess_input,
        weights_path=weights_path,
        display=False,
    )
    print_evaluation_summary(result)
