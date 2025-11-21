# ArteriAI - Coronary Heart Disease Detection System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Django](https://img.shields.io/badge/Django-4.2+-green.svg)](https://www.djangoproject.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.9+-orange.svg)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

ArteriAI is a comprehensive web-based application designed to assist in the detection and analysis of coronary heart disease from medical images. The system combines advanced deep learning models with a user-friendly web interface to provide accurate diagnosis predictions and risk assessments.

## 🏥 Features

### Core Functionality
- **Medical Image Analysis**: Support for DICOM, JPEG, and PNG medical images
- **Multi-Model Ensemble**: Uses EfficientNetB0, ResNet50, and VGG16 for improved accuracy
- **Real-time Processing**: Asynchronous image processing with progress tracking
- **Batch Processing**: Handle large datasets with batch upload and processing
- **Quality Metrics**: PSNR, MSE, SSIM, and UQI calculations for processed images

### Web Interface
- **Intuitive Upload System**: Drag-and-drop interface for medical images
- **Progress Tracking**: Real-time processing status and progress indicators
- **Result Visualization**: Detailed analysis results with confidence scores
- **Symptom Checker**: Interactive questionnaire for risk assessment
- **Responsive Design**: Modern UI built with Tailwind CSS

### Advanced Features
- **DICOM Support**: Full DICOM file handling with metadata extraction
- **Image Enhancement**: Advanced preprocessing pipeline for optimal results
- **Online Learning**: Continuous model improvement through user interactions

## 🏗️ Architecture

### Project Structure
```
CoronaryHeart/
├── arteriai/                 # Django project settings
├── preprocessing_app/        # Main Django application
│   ├── models.py            # Database models
│   ├── views.py             # Web views and API endpoints
│   ├── forms.py             # Form definitions
│   ├── symptom_checker.py   # Risk assessment module
│   ├── adversarial_defense/ # Security features
│   └── templates/           # HTML templates
├── model_training/          # Training scripts
│   ├── train_efficientnet.py
│   ├── train_resnet50.py
│   ├── train_vgg16.py
│   ├── ensemble_evaluate.py
│   ├── direct_feature_extraction.py
│   ├── recall_sensitivity_analysis.py
│   └── crop_image.py
├── models/                  # Trained model files
├── data/                    # Dataset storage
│   ├── splits/             # Train/val/test splits
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── train_labels_filtered.csv
├── processing_scripts/      # Image processing utilities
├── extracted_features_direct/ # Feature extraction results
├── recall_sensitivity_results/ # Recall analysis results
├── ensemble_results/        # Model evaluation results
└── staticfiles/            # Static assets
```

### Technology Stack
- **Backend**: Django 4.2, Python 3.8+
- **Frontend**: HTML5, CSS3, JavaScript, Tailwind CSS
- **Machine Learning**: TensorFlow 2.9+, scikit-learn
- **Image Processing**: OpenCV, scikit-image, Pillow
- **Database**: SQLite (development), PostgreSQL (production ready)
- **Medical Imaging**: PyDICOM for DICOM file handling

## 🚀 Installation

### Prerequisites
- Python 3.8 or 3.9 (for TensorFlow 2.9.x compatibility)
- pip package manager
- Virtual environment (recommended)

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd CoronaryHeart
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Node.js dependencies (for Tailwind CSS)**
   ```bash
   npm install
   ```

5. **Run database migrations**
   ```bash
   python manage.py makemigrations
   python manage.py migrate
   ```

6. **Run the development server**
   ```bash
   python manage.py runserver
   ```

## 📊 Project Functions

### 1. **Model Training Pipeline**
The system includes a comprehensive training pipeline for multiple deep learning models:

#### **Individual Model Training**
- **VGG16 Training** (`train_vgg16.py`): Trains VGG16 model with transfer learning
- **ResNet50 Training** (`train_resnet50.py`): Trains ResNet50 model with custom architecture
- **EfficientNet Training** (`train_efficientnet.py`): Trains EfficientNetB0 for optimal performance

#### **Ensemble Evaluation** (`ensemble_evaluate.py`)
- Combines predictions from all three models
- Calculates comprehensive metrics including:
  - **Accuracy**: Overall prediction accuracy
  - **Precision**: True positives / (True positives + False positives)
  - **Sensitivity (Recall)**: True positives / (True positives + False negatives)
  - **Specificity**: True negatives / (True negatives + False positives)
  - **F1-Score**: Harmonic mean of precision and recall
  - **ROC-AUC**: Area under the Receiver Operating Characteristic curve

### 2. **Data Processing Pipeline**

#### **Image Cropping** (`crop_image.py`)
- Processes original images with bounding box annotations
- Extracts stenosis regions with 110-pixel margin for context
- Creates balanced dataset with positive (stenosis) and negative (normal) samples
- Resizes all images to 224x224 pixels for consistency

#### **Dataset Splitting** (`dataset_split.py`)
- Splits data into train/validation/test sets
- Maintains class balance across splits
- Organizes data into structured directories

### 3. **Feature Extraction System**

#### **Direct Feature Extraction** (`direct_feature_extraction.py`)
Extracts comprehensive features from processed medical images:

**Geometric Features:**
- Image dimensions, area, aspect ratio, perimeter
- Contour analysis (area, perimeter, circularity, solidity)
- Shape features (eccentricity, extent)

**Texture Features:**
- Local Binary Pattern (LBP) statistics
- Gray Level Co-occurrence Matrix (GLCM) properties
- Gabor filter responses

**Intensity Features:**
- Basic statistics (mean, std, min, max, variance)
- Histogram features (entropy, energy, uniformity)
- Percentiles (10th, 25th, 50th, 75th, 90th)
- Color channel statistics (RGB)

**HOG Features:**
- Histogram of Oriented Gradients
- Summary statistics (mean, std, entropy, energy)

**Edge Features:**
- Canny edge density and count
- Sobel edge statistics
- Laplacian statistics

### 4. **Recall Sensitivity Analysis**

#### **Medical-Focused Analysis** (`recall_sensitivity_analysis.py`)
Specialized analysis for medical applications where missing positive cases is critical:

**Threshold Analysis:**
- Tests classification thresholds from 0.1 to 0.9
- Shows how recall changes with threshold
- Helps find optimal threshold for medical applications
- Creates Precision-Recall curves

**Noise Analysis:**
- Adds Gaussian noise (σ = 0.01 to 0.2)
- Measures recall degradation with noise
- Important for real-world image quality variations

**Blur Analysis:**
- Applies Gaussian blur (σ = 0 to 5)
- Tests performance on blurry images
- Critical for medical imaging where focus varies

**Brightness Analysis:**
- Adjusts image brightness (0.5x to 1.5x)
- Tests performance under different lighting conditions
- Important for consistent diagnosis

### 5. **Web Application**

#### **Django Web Interface** (`preprocessing_app/`)
- **Image Upload**: Support for DICOM, JPEG, PNG formats
- **Real-time Processing**: Asynchronous processing with progress tracking
- **Result Display**: Confidence scores and detailed analysis
- **Symptom Checker**: Interactive risk assessment questionnaire
- **Batch Processing**: Handle multiple images simultaneously

#### **Adversarial Defense** (`preprocessing_app/adversarial_defense/`)
- Protection against adversarial attacks
- Robust model ensemble
- Input validation and sanitization

## 📈 Analysis Results

### **Extracted Features Analysis** (`extracted_features_direct/`)

The feature extraction system has analyzed **1,523 images** with **73 features** per image:

**Dataset Composition:**
- **Stenosis (Positive)**: 810 images (53.2%)
- **Normal (Negative)**: 713 images (46.8%)

**Key Feature Differences:**
- **Mean Intensity**: Stenosis (129.04) vs Normal (110.94)
- **Standard Deviation**: Stenosis (50.88) vs Normal (37.66)
- **Edge Density**: Stenosis (0.309) vs Normal (0.290)
- **GLCM Contrast**: Stenosis (7.43) vs Normal (5.05)

**Feature Importance:**
- Top discriminative features identified through statistical analysis
- Feature importance ranking for model interpretability
- Class-wise feature distributions

### **Recall Sensitivity Results** (`recall_sensitivity_results/`)

**Baseline Model Performance:**
- **VGG16**: 97.04% recall
- **ResNet50**: 89.14% recall
- **EfficientNet**: 93.21% recall
- **Ensemble**: 95.80% recall

**Threshold Optimization:**
- **Optimal Threshold**: 0.10 for maximum recall
- **VGG16**: 99.88% recall at threshold 0.10
- **Ensemble**: 100% recall at threshold 0.10

**Robustness Analysis:**
- **Noise Tolerance**: Models tested with varying noise levels
- **Blur Resistance**: Performance under different blur conditions
- **Brightness Adaptation**: Performance under different lighting

## 🔧 Usage

### **Running Feature Extraction**
```bash
python run_direct_feature_extraction.py
```

### **Running Recall Analysis**
```bash
python run_recall_analysis.py
```

### **Running Ensemble Evaluation**
```bash
cd model_training
python ensemble_evaluate.py
```

### **Training Individual Models**
```bash
cd model_training
python train_vgg16.py
python train_resnet50.py
python train_efficientnet.py
```

### **Web Application**
```bash
python manage.py runserver
```
Visit `http://localhost:8000` to access the web interface.

## 📋 Requirements

See `requirements.txt` for complete Python dependencies:

```
Django>=4.2.0
TensorFlow>=2.9.0
scikit-learn>=1.0.0
opencv-python>=4.5.0
scikit-image>=0.19.0
Pillow>=8.0.0
pydicom>=2.3.0
matplotlib>=3.5.0
seaborn>=0.11.0
pandas>=1.3.0
numpy>=1.21.0
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏥 Medical Disclaimer

This system is designed for research and educational purposes. It should not be used as a substitute for professional medical diagnosis. Always consult with qualified healthcare professionals for medical decisions.

## 📞 Support

For questions and support, please open an issue in the GitHub repository or contact the development team. 