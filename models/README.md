# Placeholder for saved model weights

# Models Directory

This directory stores the trained model weights and checkpoints generated during the Neural Architecture Search experiments.

## Contents

- **Model Weights**: `.pth` files containing trained model state dictionaries
- **Best Models**: Selected architectures from NAS experiments
- **Checkpoints**: Intermediate training checkpoints

## File Naming Convention

- `CNNArchitecture_{timestamp}.pth` - General architecture models
- `best_model_naswot.pth` - Best model selected by NASWOT score
- `best_model_arch.pth` - Best model from Bayesian optimization
- `{model_name}_{experiment}.pth` - Specific experiment models

## Loading Models

```python
import torch
from src.models import CNNArchitecture

# Load model architecture
hyperparams = {...}  # Your model hyperparameters
model = CNNArchitecture(**hyperparams)

# Load trained weights
model.load_state_dict(torch.load('models/best_model.pth'))
model.eval()
```

## Model Information

Each trained model corresponds to specific hyperparameters found during the NAS process. Refer to the experiment logs or notebook outputs for the exact configuration of each saved model.

## File Naming Convention

- `best_model_naswot.pth` - Best model found using NAS-WOT metric
- `best_model_arch.pth` - Best model found using architecture scoring
- `experiment_{name}_{timestamp}.pth` - Models from specific experiments
- `checkpoint_epoch_{epoch}.pth` - Training checkpoints

## Model Information

Each saved model should be accompanied by:
- Model architecture parameters (JSON file)
- Training configuration
- Performance metrics
- Hardware efficiency metrics (FLOPs, parameters, model size)

## Usage

```python
import torch
from src.models import CNNArchitecture

# Load model
model = CNNArchitecture(**model_params)
model.load_state_dict(torch.load('models/best_model.pth'))
model.eval()
```
