"""
CNN Architecture Module

This module contains the CNN architecture class used in the NAS process.
Based on the implementation in 4A.ipynb.
"""

import torch
import torch.nn as nn


class CNNArchitecture(nn.Module):
    """
    Configurable CNN architecture for Visual Wake Words classification.
    
    This class implements the exact architecture used in the 4A notebook
    with configurable hyperparameters for Neural Architecture Search.
    """
    
    def __init__(self, **hyperparams):
        """
        Initialize CNN architecture with given hyperparameters.
        
        Args:
            **hyperparams: Dictionary containing architecture hyperparameters
        """
        super(CNNArchitecture, self).__init__()
        
        self.hyperparams = hyperparams
        
        # Fixed initial layers (as in notebook)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        # Configurable layers 4-6
        self.conv4 = nn.Conv2d(
            64, hyperparams.get("number_of_filters4", 64),
            kernel_size=hyperparams.get("kernel_size4", 3),
            stride=1,
            padding=hyperparams.get("padding4", 1)
        )
        
        self.conv5 = nn.Conv2d(
            hyperparams.get("number_of_filters4", 64),
            hyperparams.get("number_of_filters5", 128),
            kernel_size=hyperparams.get("kernel_size5", 3),
            stride=1,
            padding=hyperparams.get("padding5", 1)
        )
        
        self.conv6 = nn.Conv2d(
            hyperparams.get("number_of_filters5", 128),
            hyperparams.get("number_of_filters6", 256),
            kernel_size=hyperparams.get("kernel_size6", 3),
            stride=1,
            padding=hyperparams.get("padding6", 1)
        )
        
        # Final convolutional layer
        self.conv10 = nn.Conv2d(
            hyperparams.get("number_of_filters6", 256),
            hyperparams.get("number_of_filters10", 128),
            kernel_size=3,
            stride=1,
            padding=1
        )
        
        # Activation and regularization layers
        self.relu = nn.ReLU()
        
        # Dropout layers
        self.dropout4 = nn.Dropout2d(hyperparams.get("dropout_rate4", 0.2))
        self.dropout5 = nn.Dropout2d(hyperparams.get("dropout_rate5", 0.2))
        self.dropout6 = nn.Dropout2d(hyperparams.get("dropout_rate6", 0.2))
        
        # Max pooling layers with configurable parameters
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.maxpool4 = nn.MaxPool2d(
            kernel_size=hyperparams.get("max_pooling_kernel_size4", 2),
            stride=hyperparams.get("max_pooling_stride4", 2)
        )
        self.maxpool5 = nn.MaxPool2d(
            kernel_size=hyperparams.get("max_pooling_kernel_size5", 2),
            stride=hyperparams.get("max_pooling_stride5", 2)
        )
        self.maxpool6 = nn.MaxPool2d(
            kernel_size=hyperparams.get("max_pooling_kernel_size6", 2),
            stride=hyperparams.get("max_pooling_stride6", 2)
        )
        
        # Adaptive pooling for final layer
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers based on configuration
        self.fc_dropout = nn.Dropout(hyperparams.get("dropout_rate_fc", 0.5))
        
        num_fc_layers = hyperparams.get("number_of_fc_layers", 2)
        neurons_per_layer = hyperparams.get("number_of_neurons_per_fc_layer", 64)
        final_filters = hyperparams.get("number_of_filters10", 128)
        
        if num_fc_layers == 1:
            self.fc = nn.Linear(final_filters, 2)
        elif num_fc_layers == 2:
            self.fc1 = nn.Linear(final_filters, neurons_per_layer)
            self.fc2 = nn.Linear(neurons_per_layer, 2)
        else:  # 3 layers
            self.fc1 = nn.Linear(final_filters, neurons_per_layer)
            self.fc2 = nn.Linear(neurons_per_layer, neurons_per_layer)
            self.fc3 = nn.Linear(neurons_per_layer, 2)
        
        self.num_fc_layers = num_fc_layers
        
    def forward(self, x):
        """Forward pass through the network."""
        # Fixed initial layers
        x = self.relu(self.conv1(x))
        x = self.maxpool1(x)
        
        x = self.relu(self.conv2(x))
        x = self.maxpool2(x)
        
        x = self.relu(self.conv3(x))
        x = self.maxpool3(x)
        
        # Configurable layers 4-6
        x = self.relu(self.conv4(x))
        x = self.dropout4(x)
        x = self.maxpool4(x)
        
        x = self.relu(self.conv5(x))
        x = self.dropout5(x)
        x = self.maxpool5(x)
        
        x = self.relu(self.conv6(x))
        x = self.dropout6(x)
        x = self.maxpool6(x)
        
        # Final convolutional layer
        x = self.relu(self.conv10(x))
        x = self.adaptive_pool(x)
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        if self.num_fc_layers == 1:
            x = self.fc(x)
        elif self.num_fc_layers == 2:
            x = self.relu(self.fc1(x))
            x = self.fc_dropout(x)
            x = self.fc2(x)
        else:  # 3 layers
            x = self.relu(self.fc1(x))
            x = self.fc_dropout(x)
            x = self.relu(self.fc2(x))
            x = self.fc_dropout(x)
            x = self.fc3(x)
        
        return x


def create_model_from_params(params):
    """
    Create a CNN model from parameter dictionary.
    
    Args:
        params (dict): Dictionary containing model hyperparameters
        
    Returns:
        CNNArchitecture: Configured CNN model
    """
    return CNNArchitecture(**params)


def check_hyperparameters(params):
    """
    Check if hyperparameters satisfy the constraints from the notebook.
    
    Args:
        params (dict): Dictionary of hyperparameters
        
    Returns:
        bool: True if all constraints are satisfied
    """
    violated_conditions = []

    # Condition 1: Filter sizes should be non-decreasing (4->5->6)
    if params["number_of_filters5"] < params["number_of_filters4"]:
        violated_conditions.append("Condition 1 (Layer 5)")
    if params["number_of_filters6"] < params["number_of_filters5"]:
        violated_conditions.append("Condition 1 (Layer 6)")

    # Condition 2: Kernel sizes should be 3 or 5
    for i in range(4, 7):
        kernel_size = params[f"kernel_size{i}"]
        if kernel_size not in [3, 5]:
            violated_conditions.append(f"Condition 2 (Layer {i})")

    # Condition 3: Padding should be 1 or 2
    for i in range(4, 7):
        padding = params[f"padding{i}"]
        if padding not in [1, 2]:
            violated_conditions.append(f"Condition 3 (Layer {i})")

    # Condition 4: Dropout rates validation
    dropout_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    for i in range(4, 7):
        dropout_rate = params[f"dropout_rate{i}"]
        if dropout_rate not in dropout_rates:
            violated_conditions.append(f"Condition 4 (Layer {i})")

    # Condition 5: Max pooling parameters
    for i in range(4, 7):
        max_pooling_kernel_size = params[f"max_pooling_kernel_size{i}"]
        max_pooling_stride = params[f"max_pooling_stride{i}"]
        if max_pooling_kernel_size not in [2, 3] or max_pooling_stride not in [1, 2]:
            violated_conditions.append(f"Condition 5 (Layer {i})")

    # Condition 6: Final layer filters
    filters = [32, 64, 128, 256, 512]
    if params["number_of_filters10"] not in filters:
        violated_conditions.append("Condition 6")

    # Condition 7: Number of FC layers
    fc_layers = [1, 2, 3]
    if params["number_of_fc_layers"] not in fc_layers:
        violated_conditions.append("Condition 7")

    # Condition 8: FC neurons
    fc_neurons = [16, 32, 64, 128]
    if params["number_of_neurons_per_fc_layer"] not in fc_neurons:
        violated_conditions.append("Condition 8")

    # Condition 9: FC dropout
    fc_dropout = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    if params["dropout_rate_fc"] not in fc_dropout:
        violated_conditions.append("Condition 9")

    if violated_conditions:
        return False

    return True


def generate_hyperparameters(search_space):
    """
    Generate random hyperparameters from search space (from notebook).
    
    Args:
        search_space (dict): Search space definition
        
    Returns:
        dict: Random hyperparameters
    """
    import random
    
    hyperparameters = {
        "number_of_filters4": random.randint(*search_space["number_of_filters4"]),
        "number_of_filters5": random.randint(*search_space["number_of_filters5"]),
        "number_of_filters6": random.randint(*search_space["number_of_filters6"]),
        "kernel_size4": random.choice(search_space["kernel_size4"]),
        "kernel_size5": random.choice(search_space["kernel_size5"]),
        "kernel_size6": random.choice(search_space["kernel_size6"]),
        "padding4": random.choice(search_space["padding4"]),
        "padding5": random.choice(search_space["padding5"]),
        "padding6": random.choice(search_space["padding6"]),
        "dropout_rate4": random.choice(search_space["dropout_rate4"]),
        "dropout_rate5": random.choice(search_space["dropout_rate5"]),
        "dropout_rate6": random.choice(search_space["dropout_rate6"]),
        "max_pooling_kernel_size4": random.choice(search_space["max_pooling_kernel_size4"]),
        "max_pooling_kernel_size5": random.choice(search_space["max_pooling_kernel_size5"]),
        "max_pooling_kernel_size6": random.choice(search_space["max_pooling_kernel_size6"]),
        "max_pooling_stride4": random.choice(search_space["max_pooling_stride4"]),
        "max_pooling_stride5": random.choice(search_space["max_pooling_stride5"]),
        "max_pooling_stride6": random.choice(search_space["max_pooling_stride6"]),
        "dropout_rate_fc": random.choice(search_space["dropout_rate_fc"]),
        "number_of_filters10": random.choice(search_space["number_of_filters10"]),
        "number_of_fc_layers": random.choice(search_space["number_of_fc_layers"]),
        "number_of_neurons_per_fc_layer": random.choice(search_space["number_of_neurons_per_fc_layer"]),
    }
    return hyperparameters


# Search space definition from notebook
SEARCH_SPACE = {
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
