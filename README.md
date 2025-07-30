# Neural Architecture Search for Tiny Visual Wake Words (NAS-TVWW)

This project implements Neural Architecture Search (NAS) using Bayesian optimization and NASWOT scoring to find optimal CNN architectures for Visual Wake Words classification on resource-constrained devices.

## 🎯 Project Overview

Visual Wake Words is a computer vision task that involves detecting the presence of persons in images. This project focuses on automatically finding efficient neural network architectures that can run on tiny devices while maintaining good accuracy through:

- **Neural Architecture Search**: Automated search for optimal CNN architectures using random sampling
- **NASWOT Scoring**: Neural Architecture Search Without Training for efficient model evaluation
- **Bayesian Optimization**: Multi-objective optimization balancing accuracy, model size, and computational efficiency
- **Hardware-aware Design**: Considers FLOPs, parameters, and memory constraints for edge deployment

## 🏗️ Architecture Search Space

The search space includes configurable parameters for:

### Convolutional Layers (Layers 4-6)
- **Number of filters**: 16-256 filters per layer
- **Kernel sizes**: 3×3 or 5×5 kernels
- **Padding**: 1 or 2 pixels
- **Dropout rates**: 0.0-0.5 for regularization
- **Max pooling**: 2×2 or 3×3 kernels with different strides

### Fully Connected Layers
- **Number of layers**: 1-3 FC layers
- **Neurons per layer**: 16-128 neurons
- **Dropout rates**: 0.0-0.5 for regularization

## 📊 Dataset

- **Dataset**: Visual Wake Words (VWW) - derived from COCO dataset
- **Task**: Binary classification (person/no-person)
- **Classes**: 2 (Person present vs. Person absent)
- **Image size**: 224×224 pixels (resized from original)
- **Preprocessing**: Data augmentation (horizontal flip, rotation, color jitter) and normalization

## 🚀 Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

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
