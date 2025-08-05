"""
NASWOT (Neural Architecture Search Without Training) Implementation
Extracted from 4A.ipynb notebook
"""

import torch
import torch.nn as nn


def compute_naswot(net, inputs, device=None):
    """
    Compute NASWOT score for a neural network architecture
    
    Args:
        net: PyTorch neural network model
        inputs: Input tensor for forward pass
        device: Computing device (cuda/cpu)
    
    Returns:
        float: NASWOT score
    """
    if device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    with torch.no_grad():
        cs = list()

        def naswot_hook(module, module_input, module_output):
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

        full_code = torch.cat(cs, dim=1)

        del cs, _
        full_code_float = full_code.float()
        k = full_code_float @ full_code_float.t()
        del full_code_float
        not_full_code_float = torch.logical_not(full_code).float()
        k += not_full_code_float @ not_full_code_float.t()
        naswot_score = torch.slogdet(k).logabsdet.item()
        
        if naswot_score == float('-inf'):
            naswot_score = 1e-6

        return naswot_score


def extract_model_info(model, input_shape=(3, 96, 96)):
    """
    Extract architectural information from a model
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape
    
    Returns:
        dict: Model architecture information
    """
    model.eval()
    
    number_of_filters = []
    kernel_sizes = []
    number_of_fc_layers = 0
    total_flops = 0
    total_params = 0
    width = 0
    depth = 0
    receptive_field_size = 1
    number_of_conv_layers = 0
    current_shape = input_shape

    def calculate_output_shape(input_shape, kernel_size, padding, stride):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(stride, int):
            stride = (stride, stride)
        
        h_out = (input_shape[0] + 2 * padding[0] - kernel_size[0]) // stride[0] + 1
        w_out = (input_shape[1] + 2 * padding[1] - kernel_size[1]) // stride[1] + 1
        return (h_out, w_out)

    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            number_of_conv_layers += 1
            number_of_filters.append(layer.out_channels)
            kernel_sizes.append(layer.kernel_size)
            
            receptive_field_size += (layer.kernel_size[0] - 1)
            
            output_shape = calculate_output_shape(
                current_shape[1:],
                layer.kernel_size,
                layer.padding,
                layer.stride
            )
            
            flops = (layer.kernel_size[0] * layer.kernel_size[1] *
                     layer.in_channels * layer.out_channels *
                     output_shape[0] * output_shape[1])
            total_flops += flops

            total_params += sum(p.numel() for p in layer.parameters())
            width = max(width, layer.out_channels)
            depth += 1
            current_shape = (layer.out_channels, *output_shape)

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


def normalize(score, min_val, max_val, epsilon=1e-9):
    """Normalize a score between min_val and max_val"""
    return (score - min_val) / (max_val - min_val + epsilon)


def model_score(model_info, flops_weight, params_weight, fc_layers_weight,
                depth_weight, width_weight, receptive_field_weight, conv_layers_weight,
                min_flops, max_flops, min_params, max_params, min_fc_layers, max_fc_layers,
                min_depth, max_depth, min_width, max_width, min_receptive_field, max_receptive_field,
                min_conv_layers, max_conv_layers):
    """Calculate weighted score of a model given its information and weights"""
    
    flops_score = normalize(model_info["FLOPs"], min_flops, max_flops)
    params_score = normalize(model_info["Parameters"], min_params, max_params)
    fc_layers_score = normalize(model_info["Number of fully connected layers"], min_fc_layers, max_fc_layers)
    depth_score = normalize(model_info["Depth"], min_depth, max_depth)
    width_score = normalize(model_info["Width"], min_width, max_width)
    receptive_field_score = normalize(model_info["Receptive field size"], min_receptive_field, max_receptive_field)
    conv_layers_score = normalize(model_info["Number of convolutional layers"], min_conv_layers, max_conv_layers)

    weighted_score = (
        flops_weight * flops_score +
        params_weight * params_score +
        fc_layers_weight * fc_layers_score +
        depth_weight * depth_score +
        width_weight * width_score +
        receptive_field_weight * receptive_field_score +
        conv_layers_weight * conv_layers_score
    )

    return weighted_score
