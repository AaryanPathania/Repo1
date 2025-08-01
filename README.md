# Alzheimer's Disease Prediction - Runnable Demo

A simplified and runnable version of the Alzheimer's Disease prediction project using Convolutional Neural Networks for brain MRI image classification.

## 🧠 Overview

This project implements deep learning models to predict Alzheimer's Disease from brain MRI scans. It includes:

- **2D AlexNet**: Transfer learning approach using 2D slices from 3D MRI scans
- **3D CNN**: Direct 3D convolutional neural network processing full brain volumes
- **Synthetic Data Generator**: Creates realistic brain MRI data for demonstration
- **Complete Training Pipeline**: End-to-end training, validation, and testing

## 🚀 Quick Start

### Prerequisites

- Python 3.7 or higher
- 4GB+ RAM recommended
- 2GB+ free disk space

### Installation

1. **Clone or download the project files**:
   ```bash
   # If you have the files, navigate to the directory
   cd ad_prediction_demo
   ```

2. **Run the automated setup**:
   ```bash
   python setup.py
   ```
   This will:
   - Check system requirements
   - Install PyTorch and all dependencies
   - Verify the installation

3. **Generate demo data**:
   ```bash
   python demo_data_generator.py
   ```
   This creates synthetic brain MRI data for training and testing.

### Running the Models

**AlexNet (2D approach)**:
```bash
python ad_prediction_demo.py --mode alexnet --epochs 10 --batch_size 8
```

**3D CNN approach**:
```bash
python ad_prediction_demo.py --mode 3dcnn --epochs 5 --batch_size 4
```

**Both models**:
```bash
python ad_prediction_demo.py --mode both --epochs 5
```

## 📊 Expected Results

With the synthetic data, you should expect:
- **Training time**: 5-15 minutes per model (depending on hardware)
- **AlexNet accuracy**: 60-80% (varies with synthetic data randomness)
- **3D CNN accuracy**: 55-75% (varies with synthetic data randomness)
- **Output files**: Training plots, saved models, and performance metrics

## 📁 Project Structure

```
ad_prediction_demo/
├── ad_prediction_demo.py      # Main training script
├── demo_data_generator.py     # Synthetic data generator
├── setup.py                   # Automated setup script
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── demo_data/                 # Generated synthetic brain MRI files
├── demo_train_2classes.txt    # Training data list
├── demo_validation_2classes.txt # Validation data list
├── demo_test_2classes.txt     # Test data list
└── *.png                      # Generated plots and visualizations
```

## 🔧 Command Line Options

```bash
python ad_prediction_demo.py [OPTIONS]

Options:
  --mode {alexnet,3dcnn,both}   Model to train (default: alexnet)
  --epochs INT                  Number of training epochs (default: 10)
  --batch_size INT             Batch size for training (default: 4)
  --learning_rate FLOAT        Learning rate (default: 1e-4)
  --generate_data              Force regenerate synthetic data
  --n_samples INT              Number of samples per class (default: 30)
  --help                       Show help message
```

## 🧪 Understanding the Synthetic Data

The synthetic data generator creates realistic-looking brain MRI volumes with:

- **Normal brains**: Healthy brain structure patterns
- **AD brains**: Simulated atrophy, enlarged ventricles, cortical thinning
- **Realistic dimensions**: 121×145×121 voxels (matching real MRI data)
- **Proper formatting**: NIfTI format (.nii files) as used in medical imaging

### Data Visualization

The generator automatically creates visualizations showing:
- Axial views (horizontal slices)
- Coronal views (frontal slices)  
- Sagittal views (side slices)

## 🏥 Medical Background

### Alzheimer's Disease Detection

Alzheimer's Disease (AD) causes observable changes in brain structure:
- **Brain atrophy**: Overall brain volume reduction
- **Enlarged ventricles**: Fluid-filled spaces expand as brain tissue shrinks
- **Cortical thinning**: Outer brain layer becomes thinner
- **Hippocampal shrinkage**: Memory center shows early changes

### Deep Learning Approaches

1. **2D AlexNet Method**:
   - Extracts key 2D slices from 3D MRI volumes
   - Combines axial, coronal, and sagittal views into RGB images
   - Uses transfer learning from ImageNet-pretrained weights
   - Achieves ~86% accuracy in the original research

2. **3D CNN Method**:
   - Processes full 3D brain volumes directly
   - Captures spatial relationships in all dimensions
   - No need for slice selection or preprocessing
   - Achieved ~77% accuracy in the original research

## 🔬 Original Research

This demo is based on the research paper implementing CNN approaches for AD prediction:
- Uses ADNI (Alzheimer's Disease Neuroimaging Initiative) dataset methodology
- Implements transfer learning strategies for small medical datasets
- Compares 2D vs 3D approaches for brain image analysis

### Key Findings from Original Research:
- Transfer learning significantly improves performance on small datasets
- 2D AlexNet with carefully selected slices outperformed 3D approaches
- Proper preprocessing and normalization are crucial for medical imaging

## ⚡ Performance Tips

### For Better Training Speed:
- Use GPU if available (CUDA-enabled PyTorch will be installed on Linux)
- Reduce batch size if running out of memory
- Use fewer epochs for quick testing

### For Better Accuracy:
- Increase the number of synthetic samples: `--n_samples 100`
- Train for more epochs: `--epochs 20`
- Experiment with learning rates: `--learning_rate 1e-5`

### Memory Management:
- 3D CNN requires more memory than 2D AlexNet
- Reduce batch size for 3D: `--batch_size 2`
- Monitor system memory usage during training

## 🐛 Troubleshooting

### Common Issues:

**PyTorch installation fails**:
```bash
# Manual installation
pip install torch torchvision torchaudio
```

**Out of memory errors**:
```bash
# Reduce batch size
python ad_prediction_demo.py --batch_size 1
```

**Missing dependencies**:
```bash
# Install manually
pip install -r requirements.txt
```

**Data generation fails**:
```bash
# Check disk space and permissions
python demo_data_generator.py
```

### Getting Help:

1. Check the error messages carefully
2. Ensure all dependencies are installed: `python setup.py`
3. Try running with smaller parameters first
4. Check available system memory and disk space

## 📈 Extending the Project

### Using Real MRI Data:

To use real brain MRI data instead of synthetic data:

1. Obtain preprocessed MRI data in NIfTI format (.nii files)
2. Create data list files in the same format as the demo files:
   ```
   subject_001.nii Normal
   subject_002.nii AD
   ...
   ```
3. Update the data paths in the script
4. Ensure proper preprocessing (skull stripping, normalization, registration)

### Model Improvements:

- Implement data augmentation for better generalization
- Add more sophisticated 3D CNN architectures (ResNet3D)
- Implement ensemble methods combining 2D and 3D approaches
- Add attention mechanisms for better interpretability

### Research Extensions:

- Multi-class classification (Normal, MCI, AD)
- Longitudinal analysis (tracking disease progression)
- Explainable AI for understanding model decisions
- Integration with other biomarkers (PET, CSF, genetics)

## 📚 Dependencies

- **PyTorch**: Deep learning framework
- **nibabel**: Neuroimaging data I/O
- **scikit-image**: Image processing
- **matplotlib**: Plotting and visualization
- **numpy**: Numerical computing
- **scipy**: Scientific computing
- **tqdm**: Progress bars
- **Pillow**: Image handling
- **scikit-learn**: Machine learning utilities

## 📄 License

This project is for educational and research purposes. The original research methodologies are based on published academic work in the field of medical image analysis.

## 🙏 Acknowledgments

- Original AD prediction research methodologies
- ADNI dataset contributors (Alzheimer's Disease Neuroimaging Initiative)
- PyTorch and scientific Python community
- Medical imaging research community

## 📞 Support

For questions or issues:
1. Check this README for troubleshooting tips
2. Review the console output for specific error messages
3. Ensure all system requirements are met
4. Try the basic examples before advanced usage

---

**Note**: This is a demonstration project using synthetic data. For real medical applications, use properly validated datasets and consult with medical professionals.