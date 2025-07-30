"""
Neural Architecture Search Optimization Module

This module contains the NAS functionality including NASWOT scoring,
Bayesian optimization, and model evaluation metrics from the 4A notebook.
"""

import numpy as np
import torch
import torch.nn as nn
from bayes_opt import BayesianOptimization


def compute_naswot(net, inputs, device=None):
    """
    Compute NASWOT score for a neural network.
    
    NASWOT (Neural Architecture Search Without Training) provides a 
    training-free metric to evaluate architecture quality.
    
    Args:
        net: Neural network model
        inputs: Input tensor batch
        device: Device to run computation on
        
    Returns:
        float: NASWOT score
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with torch.no_grad():
        cs = list()

        def naswot_hook(module, module_input, module_output):
            """Hook function to capture ReLU activations."""
            code = (module_output > 0).flatten(start_dim=1)
            cs.append(code)

        hooks = list()
        for m in net.modules():
            if isinstance(m, nn.ReLU):
                hooks.append(m.register_forward_hook(naswot_hook))

        net.to(device)
        inputs = inputs.to(device)
        _ = net(inputs)

        for h in hooks:
            h.remove()

        if not cs:
            return 0.0

        full_code = torch.cat(cs, dim=1)
        del cs, _
        
        full_code_float = full_code.float()
        k = full_code_float @ full_code_float.t()
        del full_code_float
        
        eigval, _ = torch.symeig(k, eigenvectors=False)
        return torch.sum(torch.log(eigval + 1e-12)).item()


def extract_model_info(model, input_shape=(1, 3, 224, 224)):
    """
    Extract comprehensive information about a model.
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape
        
    Returns:
        dict: Model information including FLOPs, parameters, etc.
    """
    model.eval()
    
    # Initialize counters
    total_flops = 0
    total_params = 0
    number_of_filters = []
    kernel_sizes = []
    number_of_fc_layers = 0
    number_of_conv_layers = 0
    width = 0
    depth = 0
    receptive_field_size = 1
    
    current_shape = input_shape

    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            number_of_conv_layers += 1
            number_of_filters.append(layer.out_channels)
            kernel_sizes.append(layer.kernel_size)
            receptive_field_size += (layer.kernel_size[0] - 1)

            # Calculate output shape
            output_shape = calculate_output_shape(
                current_shape[2:],  # H, W
                layer.kernel_size,
                layer.padding,
                layer.stride
            )
            
            # Calculate FLOPs
            flops = (layer.kernel_size[0] * layer.kernel_size[1] *
                     layer.in_channels * layer.out_channels *
                     output_shape[0] * output_shape[1])
            total_flops += flops

            # Update total parameters
            total_params += sum(p.numel() for p in layer.parameters())

            # Update width and depth
            width = max(width, layer.out_channels)
            depth += 1

            # Update current shape
            current_shape = (current_shape[0], layer.out_channels, *output_shape)

        elif isinstance(layer, nn.Linear):
            number_of_fc_layers += 1
            total_params += sum(p.numel() for p in layer.parameters())
            depth += 1
    
    return {
        "Number of filters": number_of_filters,
        "Kernel sizes": kernel_sizes,
        "Number of fully connected layers": number_of_fc_layers,
        "FLOPs": total_flops,
        "Parameters": total_params,
        "Width": width,
        "Depth": depth,
        "Receptive field size": receptive_field_size,
        "Number of convolutional layers": number_of_conv_layers
    }


def calculate_output_shape(input_shape, kernel_size, padding, stride):
    """
    Calculate output shape after convolution operation.
    
    Args:
        input_shape: Input tensor shape (H, W)
        kernel_size: Convolution kernel size
        padding: Padding size
        stride: Stride size
        
    Returns:
        tuple: Output shape (H, W)
    """
    if isinstance(kernel_size, tuple):
        kernel_h, kernel_w = kernel_size
    else:
        kernel_h = kernel_w = kernel_size
        
    if isinstance(padding, tuple):
        pad_h, pad_w = padding
    else:
        pad_h = pad_w = padding
        
    if isinstance(stride, tuple):
        stride_h, stride_w = stride
    else:
        stride_h = stride_w = stride
    
    height, width = input_shape
    out_h = (height + 2 * pad_h - kernel_h) // stride_h + 1
    out_w = (width + 2 * pad_w - kernel_w) // stride_w + 1
    
    return (out_h, out_w)


def calculate_final_conv_output_size(input_size, hyperparams):
    """
    Calculate final convolutional output size based on hyperparameters.
    Used in the notebook for FC layer input size calculation.
    
    Args:
        input_size: Input image size
        hyperparams: Dictionary of hyperparameters
        
    Returns:
        int: Final conv output size (flattened)
    """
    output_size = input_size
    for i in range(4, 7):
        kernel_size = hyperparams[f"kernel_size{i}"]
        padding = hyperparams[f"padding{i}"]
        max_pooling_kernel_size = hyperparams[f"max_pooling_kernel_size{i}"]
        max_pooling_stride = hyperparams[f"max_pooling_stride{i}"]

        # Calculate the output size after the convolutional layer
        output_size = (output_size + 2 * padding - kernel_size) // 1 + 1

        # Calculate the output size after the max pooling layer
        output_size = (output_size - max_pooling_kernel_size) // max_pooling_stride + 1

    return output_size**2 * hyperparams["number_of_filters6"]


def normalize(score, min_val, max_val, epsilon=1e-9):
    """
    Normalize a score between min_val and max_val.
    
    Args:
        score: Score to normalize
        min_val: Minimum value
        max_val: Maximum value
        epsilon: Small value to avoid division by zero
        
    Returns:
        float: Normalized score
    """
    if max_val - min_val < epsilon:
        return 0.5
    return (score - min_val) / (max_val - min_val)


def model_score(model_info, **weights):
    """
    Calculate a composite score for a model based on multiple metrics.
    
    Args:
        model_info (dict): Model information dictionary
        **weights: Weight parameters for scoring
        
    Returns:
        float: Composite model score
    """
    # Extract metrics
    flops = model_info["FLOPs"]
    params = model_info["Parameters"]
    fc_layers = model_info["Number of fully connected layers"]
    depth = model_info["Depth"]
    width = model_info["Width"]
    receptive_field = model_info["Receptive field size"]
    conv_layers = model_info["Number of convolutional layers"]
    
    # Normalize metrics
    norm_flops = normalize(flops, weights.get('min_flops', 0), weights.get('max_flops', 1))
    norm_params = normalize(params, weights.get('min_params', 0), weights.get('max_params', 1))
    norm_fc = normalize(fc_layers, weights.get('min_fc_layers', 1), weights.get('max_fc_layers', 3))
    norm_depth = normalize(depth, weights.get('min_depth', 1), weights.get('max_depth', 20))
    norm_width = normalize(width, weights.get('min_width', 16), weights.get('max_width', 512))
    norm_receptive = normalize(receptive_field, weights.get('min_receptive_field', 1), 
                              weights.get('max_receptive_field', 100))
    norm_conv = normalize(conv_layers, weights.get('min_conv_layers', 1), 
                         weights.get('max_conv_layers', 10))
    
    # Weights for different metrics (prefer lower complexity)
    w_flops = weights.get('w_flops', 0.3)
    w_params = weights.get('w_params', 0.3)
    w_fc = weights.get('w_fc', 0.1)
    w_depth = weights.get('w_depth', 0.1)
    w_width = weights.get('w_width', 0.1)
    w_receptive = weights.get('w_receptive', 0.05)
    w_conv = weights.get('w_conv', 0.05)
    
    # Calculate composite score (lower is better for efficiency)
    score = (w_flops * (1 - norm_flops) + 
             w_params * (1 - norm_params) + 
             w_fc * (1 - norm_fc) + 
             w_depth * (1 - norm_depth) + 
             w_width * (1 - norm_width) + 
             w_receptive * norm_receptive + 
             w_conv * norm_conv)
    
    return score


def create_bayesian_optimizer(pbounds, n_init_points=5, n_iter=25):
    """
    Create and configure Bayesian optimizer for NAS.
    
    Args:
        pbounds (dict): Parameter bounds for optimization
        n_init_points (int): Number of initial random points
        n_iter (int): Number of optimization iterations
        
    Returns:
        BayesianOptimization: Configured optimizer
    """
    def objective_function(**params):
        """
        Objective function for Bayesian optimization.
        This should be replaced with actual model evaluation.
        """
        # Placeholder - in practice this would:
        # 1. Create model with params
        # 2. Evaluate with NASWOT or training
        # 3. Return score
        return np.random.random()
    
    optimizer = BayesianOptimization(
        f=objective_function,
        pbounds=pbounds,
        random_state=42,
        verbose=2
    )
    
    return optimizer


def generate_models(search_space, max_models=500, max_retries=20000, input_image_size=96):
    """
    Generate multiple CNN models using the search space from the notebook.
    
    Args:
        search_space: Dictionary defining the search space
        max_models: Maximum number of models to generate
        max_retries: Maximum retry attempts
        input_image_size: Input image size for final conv calculation
        
    Returns:
        list: List of generated models with their hyperparameters
    """
    from .models import generate_hyperparameters, check_hyperparameters, create_model_from_params
    
    models = []
    model_hyperparams = []
    
    retry_count = 0
    while len(models) < max_models and retry_count < max_retries:
        # Generate random hyperparameters
        hyperparameters = generate_hyperparameters(search_space)
        
        # Check if hyperparameters satisfy constraints
        if check_hyperparameters(hyperparameters):
            # Calculate final conv output size
            final_conv_output_size = calculate_final_conv_output_size(
                input_image_size, hyperparameters
            )
            hyperparameters["final_conv_output_size"] = final_conv_output_size
            
            # Create model
            try:
                model = create_model_from_params(hyperparameters)
                models.append(model)
                model_hyperparams.append(hyperparameters)
            except Exception as e:
                print(f"Error creating model: {e}")
        
        retry_count += 1
    
    print(f"Generated {len(models)} valid models out of {max_models} requested")
    return models, model_hyperparams


def evaluate_models_with_naswot(models, input_batch, device=None):
    """
    Evaluate multiple models using NASWOT scoring.
    
    Args:
        models: List of models to evaluate
        input_batch: Batch of input data for NASWOT computation
        device: Device to run evaluation on
        
    Returns:
        list: NASWOT scores for each model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    scores = []
    for i, model in enumerate(models):
        try:
            score = compute_naswot(model, input_batch, device)
            scores.append(score)
            if (i + 1) % 50 == 0:
                print(f"Evaluated {i + 1}/{len(models)} models")
        except Exception as e:
            print(f"Error evaluating model {i}: {e}")
            scores.append(0.0)
    
    return scores


def rank_models_by_naswot(models, naswot_scores):
    """
    Rank models by their NASWOT scores.
    
    Args:
        models: List of models
        naswot_scores: List of corresponding NASWOT scores
        
    Returns:
        tuple: (ranked_models, ranked_scores, ranking_indices)
    """
    # Sort by NASWOT score (higher is better)
    ranking_indices = sorted(range(len(naswot_scores)), 
                           key=lambda i: naswot_scores[i], reverse=True)
    
    ranked_models = [models[i] for i in ranking_indices]
    ranked_scores = [naswot_scores[i] for i in ranking_indices]
    
    return ranked_models, ranked_scores, ranking_indices
