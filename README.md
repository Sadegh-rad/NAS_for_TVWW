# Neural Architecture Search for Tiny Visual Wake Words

A PyTorch implementation of Neural Architecture Search (NAS) using NASWOT (Neural Architecture Search Without Training) for the Visual Wake Words dataset, designed for efficient tiny model deployment.

## Overview

This project implements a training-free neural architecture search method that evaluates CNN architectures without full training, making it suitable for resource-constrained environments. The approach uses NASWOT scoring combined with Bayesian optimization to find optimal architectures for binary image classification.

## Project Structure

```
NAS_for_TVWW/
├── 4A.ipynb                    # Main implementation notebook
├── nas_model.py                # CNN model and training functions
├── naswot_scoring.py           # NASWOT scoring implementation  
├── search_space.py             # Hyperparameter search space utilities
├── requirements.txt            # Dependencies
├── project_report.pdf          # Original project report
└── README.md                   # This file
```
## Key Components

### 1. CustomDNN Architecture
- Configurable CNN with 3 convolutional blocks
- Batch normalization and dropout for regularization
- Variable fully connected layers  
- Designed for 96x96 input images

### 2. NASWOT Scoring
- Training-free architecture evaluation
- Uses ReLU activation patterns to compute architecture quality
- Significantly faster than traditional training-based evaluation

### 3. Search Space
- Configurable number of filters (16, 32, 64, 128, 256)
- Kernel sizes (3x3, 5x5)
- Dropout rates (0.1, 0.2, 0.3)
- Fully connected layer configurations

### 4. Dataset Handling
- Visual Wake Words dataset from COCO
- Balanced dataset creation for binary classification
- Custom data transformations and augmentation

## Requirements

```
torch>=1.9.0
torchvision>=0.10.0
pyvww
bayesian-optimization
torchinfo
matplotlib
seaborn
tqdm
numpy
Pillow
```

## Usage

### Google Colab Setup
The main implementation is in `4A.ipynb` designed for Google Colab:

1. Mount Google Drive for dataset access
2. Install required packages  
3. Load and preprocess Visual Wake Words dataset
4. Generate candidate architectures
5. Evaluate using NASWOT scoring
6. Apply Bayesian optimization for hyperparameter search
7. Train best architectures
8. Test with camera inference

### Local Usage
You can also use the extracted modules locally:

```python
from nas_model import CustomDNN, train_and_evaluate_model
from naswot_scoring import compute_naswot
from search_space import generate_hyperparameters

# Generate hyperparameters
hyperparams = generate_hyperparameters()

# Create model
model = CustomDNN(hyperparams)

# Compute NASWOT score
sample_input = torch.randn(1, 3, 96, 96)
score = compute_naswot(model, sample_input)
```
## Key Features

- **Training-Free Evaluation**: Uses NASWOT for rapid architecture assessment
- **Bayesian Optimization**: Efficient hyperparameter search
- **Balanced Dataset**: Handles class imbalance in Visual Wake Words
- **Mobile-Friendly**: Architectures optimized for edge deployment
- **Interactive Inference**: Camera capture and real-time prediction

## Architecture Search Space

The search includes:
- **Convolutional Layers**: 3 configurable blocks
- **Filter Counts**: Progressive increase (16→32→64→128→256)
- **Kernel Sizes**: 3x3 and 5x5 convolutions  
- **Regularization**: Dropout (0.1-0.3), Batch normalization
- **Pooling**: Max pooling with stride 2
- **Fully Connected**: 2-3 layers with 32-128 neurons

## Results

The notebook demonstrates:
- Comparison between NASWOT-selected and architecture-score-selected models
- Training curves and accuracy metrics
- Real-time inference capabilities
- Model size and efficiency analysis

## Google Colab Integration

The implementation leverages Google Colab features:
- GPU acceleration for model training
- Google Drive integration for dataset storage
- Camera access for live inference testing  
- Visualization tools for results analysis

## Citation

Based on research in Neural Architecture Search and the Visual Wake Words dataset. This implementation focuses on efficient, training-free architecture evaluation suitable for edge AI applications.

### Running the Project
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Open `4A.ipynb` in Jupyter/Colab
4. Configure data paths in the notebook
5. Run the notebook cells sequentially

### Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Sadegh-rad/NAS_for_TVWW/blob/main/4A.ipynb)

## 📁 Project Structure

```
NAS_for_TVWW/
├── README.md                 # Project documentation
├── 4A.ipynb                 # Main implementation notebook
├── project_report.pdf       # Detailed project report
├── requirements.txt         # Python dependencies
├── example_usage.py         # Example usage script
├── src/                     # Source code modules
│   ├── models/             # CNN architecture definitions
│   ├── data/               # Data loading and preprocessing
│   ├── optimization/       # NAS algorithms and scoring
│   └── utils/              # Training and utility functions
├── experiments/            # Experiment configurations
└── models/                 # Saved model weights
```

## 🧠 Methodology

### 1. Architecture Generation
- Random sampling from constrained search space
- Constraint validation for architecture feasibility
- Generation of 500+ candidate architectures

### 2. Model Evaluation Metrics
- **NASWOT Score**: Training-free architecture evaluation
- **Accuracy**: Model performance on validation set
- **Model Complexity**: FLOPs, parameters, model size
- **Hardware Metrics**: Depth, width, receptive field

### 3. Multi-Objective Optimization
- Bayesian optimization for hyperparameter tuning
- Pareto front analysis for trade-off exploration
- Ranking based on composite scoring function

## 📈 Implementation Details

### Key Components

1. **CNN Architecture Class**: Configurable CNN with dynamic layer generation
2. **Data Pipeline**: Balanced dataset creation with augmentation
3. **NASWOT Scoring**: Efficient architecture evaluation without training
4. **Training Pipeline**: Full training with early stopping and learning rate scheduling
5. **Evaluation Framework**: Comprehensive model analysis and comparison

### Architecture Constraints
- Non-decreasing filter sizes across layers
- Balanced dataset with equal class representation
- Hardware-aware parameter limits
- Validation of hyperparameter combinations

## � Experiments

The notebook includes:
- Architecture search with 500+ models
- NASWOT-based ranking and selection
- Bayesian optimization for best architectures
- Training and evaluation of top models
- Performance comparison and analysis
- Visualization of results and model characteristics

## � Key Features from Implementation

- **Balanced Dataset Loading**: Equal sampling from person/no-person classes
- **Data Augmentation**: Random transforms for better generalization
- **Early Stopping**: Prevents overfitting during training
- **Model Checkpointing**: Saves best models during training
- **Comprehensive Logging**: Tracks training progress and metrics
- **Visualization Tools**: Plots for training curves and sample images

## 📧 Contact

For questions about this implementation, please open an issue in the GitHub repository.
