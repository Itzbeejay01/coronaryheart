from evaluation_utils import evaluate_model, print_evaluation_summary


MODEL_PATH = 'models/vgg16_final_model.keras'


if __name__ == "__main__":
    result = evaluate_model(
        model_label='VGG16',
        model_path=MODEL_PATH,
        preprocess_function=None,
        rescale=1.0 / 255.0,
        display=False,
    )
    print_evaluation_summary(result)

