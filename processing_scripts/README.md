# ArteriAI

ArteriAI is a web-based application designed to assist in the detection and analysis of coronary heart disease from medical images. It provides a user-friendly interface for uploading, processing, and analyzing relevant medical images using advanced deep learning models.

## How the App Works

1. **Image Upload**: Users can upload medical images (DICOM, JPEG, PNG) through the web interface.
2. **Preprocessing**: Uploaded images are preprocessed (resized, enhanced, noise-reduced) to standardize input for the models.
3. **Model Inference**: The app uses state-of-the-art deep learning models (EfficientNetB0, ResNet50, VGG16) to classify images as indicating no disease or disease.
4. **Results & Visualization**: Users receive predictions, confidence scores, and visual explanations (e.g., heatmaps) for each image.
5. **Batch Processing**: For large datasets, batch scripts (see below) can be used to process images offline.

### Main Features
- Web-based interface for easy image upload and result viewing
- Support for multiple image formats and modalities
- Robust preprocessing and quality enhancement pipeline
- Ensemble models for improved accuracy
- Visualization tools for interpretability
- Batch processing scripts for large-scale offline analysis

## Setup Instructions

### 1. Environment Setup
- Use **Python 3.8 or 3.9** (TensorFlow 2.9.x compatibility)
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### 2. Dataset Preparation
- Place your coronary heart disease image datasets in the `data/` directory, organized as follows:
  ```
  data/
    inbreast/
      no_disease/
      disease/
    cbis-ddsm/
      no_disease/
      disease/
    mri/
      no_disease/
      disease/
  ```
- Each subfolder should contain PNG/JPG images for each class.

### 3. Model Training
- Run the ensemble training script (update with your dataset path if needed):
  ```bash
  # Example (update path as needed)
  python -c "from preprocessing_app.adversarial_defense.ensemble_model import EnsembleModel; from preprocessing_app.adversarial_defense.data_loader import DataLoader; import os; data_loader = DataLoader(); x_train, y_train, x_test, y_test = data_loader.load_all_datasets(os.path.join(os.path.dirname(__file__), '../data')); model = EnsembleModel(); model.train(x_train, y_train, epochs=20, batch_size=32)"
  ```
- Model checkpoints will be saved in `model_training/checkpoints/`.

### 4. Running the Web App
- Start the Django server:
  ```bash
  python manage.py runserver
  ```
- Access the app at [http://localhost:8000](http://localhost:8000)

## Notes
- No adversarial training or ImageNet transfer learning is included.
- All references to breast cancer have been removed; this project is focused on coronary heart disease.
- For best results, use high-quality, labeled datasets.

## License
MIT 