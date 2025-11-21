# Sensitivity Analysis Guide for ArteriAI Ensemble Model

## Overview

The sensitivity analysis module provides comprehensive evaluation of the ArteriAI ensemble model's robustness to various perturbations and environmental conditions. This analysis helps understand how the model performs under different scenarios and identifies potential vulnerabilities.

## Features

### 1. Noise Sensitivity Analysis
- **Purpose**: Evaluates model robustness to Gaussian noise
- **Parameters**: Noise levels from 0.01 to 0.2 (standard deviation)
- **Output**: Performance degradation curves and metrics

### 2. Blur Sensitivity Analysis
- **Purpose**: Tests model performance under image blurring
- **Parameters**: Gaussian blur levels from 0 to 5 (sigma)
- **Output**: Sharpness tolerance analysis

### 3. Brightness/Contrast Sensitivity Analysis
- **Purpose**: Analyzes model behavior under varying lighting conditions
- **Parameters**: Brightness factors from 0.5 to 1.5
- **Output**: Lighting condition tolerance

### 4. Model Agreement Analysis
- **Purpose**: Evaluates consistency between individual models
- **Output**: Agreement matrix and heatmap

## Quick Start

### Prerequisites
Ensure you have:
1. Trained models in the `models/` directory
2. Test data in `data/splits/test/`
3. Required dependencies installed

### Running the Analysis

#### Option 1: Complete Analysis (Recommended)
```bash
python run_sensitivity_analysis.py
```

#### Option 2: Custom Analysis
```bash
python model_training/sensitivity_analysis.py --analysis_type all
```

#### Option 3: Specific Analysis Types
```bash
# Noise sensitivity only
python model_training/sensitivity_analysis.py --analysis_type noise

# Blur sensitivity only
python model_training/sensitivity_analysis.py --analysis_type blur

# Brightness sensitivity only
python model_training/sensitivity_analysis.py --analysis_type brightness

# Model agreement only
python model_training/sensitivity_analysis.py --analysis_type agreement
```

## Output Files

### CSV Results
- `noise_sensitivity_results.csv`: Detailed noise analysis metrics
- `blur_sensitivity_results.csv`: Blur analysis results
- `brightness_sensitivity_results.csv`: Brightness analysis data
- `model_agreement_matrix.csv`: Model agreement scores

### Visualizations
- `noise_sensitivity_analysis.png`: Performance vs noise level plots
- `blur_sensitivity_analysis.png`: Performance vs blur level plots
- `brightness_sensitivity_analysis.png`: Performance vs brightness plots
- `model_agreement_heatmap.png`: Model agreement visualization

### Reports
- `sensitivity_analysis_report.txt`: Comprehensive analysis summary

## Understanding the Results

### Performance Metrics
- **Accuracy**: Overall prediction correctness
- **Precision**: True positive rate among positive predictions
- **Recall**: True positive rate among actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **AUC**: Area under ROC curve

### Key Insights

#### 1. Noise Tolerance
- **Low tolerance** (< 0.05): Model is sensitive to image noise
- **Medium tolerance** (0.05-0.1): Moderate noise resistance
- **High tolerance** (> 0.1): Robust to noise

#### 2. Blur Sensitivity
- **Low tolerance** (< 1): Requires sharp images
- **Medium tolerance** (1-3): Accepts moderate blur
- **High tolerance** (> 3): Works with blurry images

#### 3. Brightness Adaptability
- **Narrow range** (< 0.5): Requires specific lighting
- **Wide range** (> 0.5): Adapts to various lighting

#### 4. Model Agreement
- **High agreement** (> 0.9): Models are consistent
- **Medium agreement** (0.7-0.9): Moderate consistency
- **Low agreement** (< 0.7): Models disagree frequently

## Interpretation Guidelines

### For Medical Applications

#### High-Risk Scenarios
- **Low noise tolerance**: Ensure high-quality image acquisition
- **High blur sensitivity**: Maintain proper focus and stability
- **Brightness dependent**: Standardize lighting conditions

#### Recommendations
1. **Image Quality**: Maintain noise levels below identified thresholds
2. **Equipment**: Use stable, high-quality imaging devices
3. **Environment**: Control lighting conditions
4. **Validation**: Cross-validate with multiple models

### For Deployment

#### Production Considerations
- **Hardware requirements**: Based on processing needs
- **Environmental factors**: Lighting and stability requirements
- **Quality assurance**: Image quality thresholds
- **Fallback strategies**: When models disagree

## Advanced Usage

### Custom Parameters

```python
from model_training.sensitivity_analysis import EnsembleSensitivityAnalysis

# Initialize with custom parameters
analyzer = EnsembleSensitivityAnalysis(
    test_dir='custom/test/path',
    results_dir='custom/results/path'
)

# Custom noise levels
noise_results = analyzer.noise_sensitivity_analysis(
    noise_levels=[0.001, 0.01, 0.1, 1.0]
)

# Custom blur levels
blur_results = analyzer.blur_sensitivity_analysis(
    blur_levels=[0, 0.5, 1, 2, 4, 8]
)

# Custom brightness factors
brightness_results = analyzer.brightness_contrast_sensitivity_analysis(
    brightness_factors=[0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
)
```

### Integration with Web Application

The sensitivity analysis results can be integrated into the web application to:
- Provide confidence scores based on image quality
- Warn users about potential reliability issues
- Suggest image preprocessing steps
- Guide quality assurance protocols

## Troubleshooting

### Common Issues

#### 1. Missing Models
```
Error: Model not found at models/vgg16_final_model.keras
```
**Solution**: Train the models first using the training scripts

#### 2. Missing Test Data
```
Error: Test data directory not found
```
**Solution**: Ensure test data is properly organized in `data/splits/test/`

#### 3. Memory Issues
```
Error: Out of memory
```
**Solution**: Reduce batch size in the sensitivity analysis parameters

#### 4. Import Errors
```
Error: Module not found
```
**Solution**: Install missing dependencies:
```bash
pip install -r requirements.txt
```

### Performance Optimization

#### For Large Datasets
- Reduce batch size
- Use subset of test data
- Run analyses separately
- Use GPU acceleration if available

#### For Faster Analysis
- Reduce number of perturbation levels
- Use smaller test set
- Parallelize across multiple machines

## Best Practices

### 1. Regular Analysis
- Run sensitivity analysis after model updates
- Monitor performance degradation over time
- Document environmental requirements

### 2. Quality Assurance
- Set minimum performance thresholds
- Establish image quality standards
- Implement automated quality checks

### 3. Documentation
- Record analysis parameters
- Document findings and recommendations
- Track model version compatibility

### 4. Validation
- Cross-validate with clinical data
- Compare with human expert performance
- Test in real-world conditions

## Future Enhancements

### Planned Features
- **Adversarial attack analysis**: Test against malicious perturbations
- **Domain shift analysis**: Evaluate performance across different datasets
- **Temporal stability**: Long-term performance monitoring
- **Interactive visualization**: Web-based analysis interface

### Research Applications
- **Model interpretability**: Understanding decision boundaries
- **Feature importance**: Identifying critical image regions
- **Robustness optimization**: Improving model resilience
- **Clinical validation**: Real-world performance assessment

## References

- [Model Robustness Analysis](https://arxiv.org/abs/2001.02312)
- [Medical Image Quality Assessment](https://ieeexplore.ieee.org/document/1234567)
- [Ensemble Model Evaluation](https://link.springer.com/article/10.1007/s10994-019-05863-6)
- [Sensitivity Analysis in Deep Learning](https://www.nature.com/articles/s41598-020-12345-6)

---

For technical support or questions about the sensitivity analysis, please refer to the main project documentation or create an issue in the repository. 