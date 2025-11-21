import os
import numpy as np
import shap
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

# SHAP for tabular models (e.g., symptom checker)
def shap_tabular(model, X, feature_names, output_dir='feature_importance'):
    os.makedirs(output_dir, exist_ok=True)
    explainer = shap.Explainer(model.predict, X)
    shap_values = explainer(X)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shap_summary.png'))
    plt.close()
    print(f"SHAP summary plot saved to {output_dir}/shap_summary.png")

# Grad-CAM for CNN models
def grad_cam(model, img_path, layer_name, output_dir='feature_importance'):
    os.makedirs(output_dir, exist_ok=True)
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    class_idx = np.argmax(preds[0])
    class_output = model.output[:, class_idx]
    last_conv_layer = model.get_layer(layer_name)

    grads = Model(inputs=model.input, outputs=[last_conv_layer.output, class_output])([x])
    conv_output, predictions = grads
    conv_output = conv_output[0]
    weights = np.mean(conv_output, axis=(0, 1))
    cam = np.dot(conv_output, weights)
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    cam = np.uint8(255 * cam)
    cam = np.expand_dims(cam, axis=2)
    cam = np.repeat(cam, 3, axis=2)
    cam = image.array_to_img(cam)
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.imshow(cam, cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, f'grad_cam_{os.path.basename(img_path)}'))
    plt.close()
    print(f"Grad-CAM saved to {output_dir}/grad_cam_{os.path.basename(img_path)}")

# Example usage (uncomment and adapt as needed):
# shap_tabular(model, X, feature_names)
# grad_cam(cnn_model, 'path/to/image.png', 'block5_conv3') 