"""
Hyperparameter Search Space and Utilities
Extracted from 4A.ipynb notebook
"""

import random
import itertools


def check_hyperparameters(params):
    """Check if hyperparameters satisfy search space constraints"""
    violated_conditions = []

    # Condition 1: Filters should increase or stay the same
    prev_filters = params["number_of_filters4"]
    for i in range(5, 7):
        curr_filters = params[f"number_of_filters{i}"]
        if curr_filters < prev_filters:
            violated_conditions.append(f"Condition 1 (Layer {i})")
        prev_filters = curr_filters

    # Condition 2: Kernel sizes should be 3 or 5
    for i in range(4, 7):
        kernel_size = params[f"kernel_size{i}"]
        if kernel_size not in (3, 5):
            violated_conditions.append(f"Condition 2 (Layer {i})")

    # Condition 3: Padding should be 1 or 2
    for i in range(4, 7):
        padding = params[f"padding{i}"]
        if padding not in (1, 2):
            violated_conditions.append(f"Condition 3 (Layer {i})")

    # Condition 4: Dropout rates
    dropout_rates = [0.1, 0.2, 0.3]
    for i in range(4, 7):
        dropout_rate = params[f"dropout_rate{i}"]
        if dropout_rate not in dropout_rates:
            violated_conditions.append(f"Condition 4 (Layer {i})")

    # Condition 5: Max pooling parameters
    for i in range(4, 7):
        max_pooling_kernel_size = params[f"max_pooling_kernel_size{i}"]
        max_pooling_stride = params[f"max_pooling_stride{i}"]
        if max_pooling_kernel_size not in (2, 3) or max_pooling_stride != 2:
            violated_conditions.append(f"Condition 5 (Layer {i})")

    # Condition 6: Final layer filters
    filters = [64, 128, 256]
    if params["number_of_filters10"] not in filters:
        violated_conditions.append("Condition 6")

    # Condition 7: Number of FC layers
    fc_layers = [2, 3]
    if params["number_of_fc_layers"] not in fc_layers:
        violated_conditions.append("Condition 7")

    # Condition 8: FC layer neurons
    fc_neurons = [32, 64, 96, 128]
    if params["number_of_neurons_per_fc_layer"] not in fc_neurons:
        violated_conditions.append("Condition 8")

    # Condition 9: FC dropout rate
    if params["dropout_rate_fc"] not in [0.1, 0.2, 0.3]:
        violated_conditions.append("Condition 9")

    return violated_conditions


def generate_hyperparameters():
    """Generate random hyperparameters within the search space"""
    hyperparams = {}
    
    # Convolutional layers (4, 5, 6)
    for i in range(4, 7):
        hyperparams[f"number_of_filters{i}"] = random.choice([16, 32, 64, 128])
        hyperparams[f"kernel_size{i}"] = random.choice([3, 5])
        hyperparams[f"padding{i}"] = random.choice([1, 2])
        hyperparams[f"dropout_rate{i}"] = random.choice([0.1, 0.2, 0.3])
        hyperparams[f"max_pooling_kernel_size{i}"] = random.choice([2, 3])
        hyperparams[f"max_pooling_stride{i}"] = 2
    
    # Final convolutional layer
    hyperparams["number_of_filters10"] = random.choice([64, 128, 256])
    
    # Fully connected layers
    hyperparams["number_of_fc_layers"] = random.choice([2, 3])
    hyperparams["number_of_neurons_per_fc_layer"] = random.choice([32, 64, 96, 128])
    hyperparams["dropout_rate_fc"] = random.choice([0.1, 0.2, 0.3])
    
    # Calculate final conv output size (this needs to be computed based on input size)
    # For 96x96 input after conv and pooling operations
    hyperparams["final_conv_output_size"] = 256  # This should be calculated dynamically
    
    return hyperparams


def create_balanced_dataset(dataset, target_class=1):
    """Create balanced dataset by matching minority class count"""
    target_indices = []
    non_target_indices = []
    
    for i, (_, target) in enumerate(dataset):
        if target == target_class:
            target_indices.append(i)
        else:
            non_target_indices.append(i)
    
    # Balance by matching smaller class size
    min_count = min(len(target_indices), len(non_target_indices))
    balanced_indices = (
        random.sample(target_indices, min_count) + 
        random.sample(non_target_indices, min_count)
    )
    
    return balanced_indices
