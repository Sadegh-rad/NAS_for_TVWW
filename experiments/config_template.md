# NAS Experiment Configurations

This file contains the configuration parameters used in the 4A notebook experiments.

## Architecture Search Space (from notebook)

```python
search_space = {
    "number_of_filters4": (16, 64),
    "number_of_filters5": (32, 128), 
    "number_of_filters6": (64, 256),
    "kernel_size4": [3, 5],
    "kernel_size5": [3, 5],
    "kernel_size6": [3, 5],
    "padding4": [1, 2],
    "padding5": [1, 2],
    "padding6": [1, 2],
    "dropout_rate4": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    "dropout_rate5": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    "dropout_rate6": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    "max_pooling_kernel_size4": [2, 3],
    "max_pooling_kernel_size5": [2, 3], 
    "max_pooling_kernel_size6": [2, 3],
    "max_pooling_stride4": [1, 2],
    "max_pooling_stride5": [1, 2],
    "max_pooling_stride6": [1, 2],
    "dropout_rate_fc": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    "number_of_filters10": [32, 64, 128, 256, 512],
    "number_of_fc_layers": [1, 2, 3],
    "number_of_neurons_per_fc_layer": [16, 32, 64, 128],
}
```

## Model Generation Parameters

- **MAX_MODELS**: 500 (target number of models to generate)
- **MAX_RETRIES**: 20,000 (maximum attempts to generate valid models)
- **INPUT_IMAGE_SIZE**: 96 (for final conv output size calculation)

## Architecture Constraints

1. **Filter Progression**: Filters should be non-decreasing (layer 5 ≥ layer 4, layer 6 ≥ layer 5)
2. **Kernel Sizes**: Only 3×3 or 5×5 kernels allowed
3. **Padding**: Only 1 or 2 pixel padding
4. **Dropout Rates**: Range from 0.0 to 0.5 in increments of 0.1
5. **Max Pooling**: Kernel sizes 2×2 or 3×3, strides 1 or 2
6. **Final Layer Filters**: Must be one of [32, 64, 128, 256, 512]
7. **FC Layers**: 1, 2, or 3 fully connected layers
8. **FC Neurons**: 16, 32, 64, or 128 neurons per layer
9. **FC Dropout**: Range from 0.0 to 0.5

## Training Configuration

```python
training_config = {
    "num_epochs": 40,
    "batch_size": 32,
    "learning_rate": 0.01,
    "optimizer": "SGD",
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "scheduler": "ReduceLROnPlateau",
    "patience": 10,
    "factor": 0.1,
    "early_stopping_patience": 10
}
```

## Data Configuration

```python
data_config = {
    "image_size": (224, 224),
    "batch_size": 32,
    "num_workers": 2,
    "augmentation": {
        "horizontal_flip": 0.5,
        "rotation": 10,
        "color_jitter": {
            "brightness": 0.2,
            "contrast": 0.2, 
            "saturation": 0.2,
            "hue": 0.1
        }
    },
    "normalization": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225]
    }
}
```

## NASWOT Evaluation

- **Input Batch Size**: 32 samples for NASWOT computation
- **Hook Target**: ReLU activation layers
- **Score Calculation**: Sum of log eigenvalues of activation correlation matrix

## Bayesian Optimization

```python
bayesian_config = {
    "n_init_points": 10,
    "n_iter": 50,
    "acquisition_function": "expected_improvement",
    "kappa": 2.576,
    "xi": 0.0
}
```

## Model Evaluation Metrics

1. **NASWOT Score**: Training-free architecture ranking
2. **Accuracy**: Validation set performance
3. **Model Complexity**: Parameter count and FLOPs
4. **Hardware Metrics**: Model size, depth, width
5. **Composite Score**: Weighted combination of metrics

## Results Analysis

- **Top Models**: Selected based on NASWOT ranking
- **Performance Comparison**: Train/validation accuracy curves
- **Efficiency Analysis**: FLOPs vs accuracy trade-offs
- **Architecture Insights**: Impact of different hyperparameters
