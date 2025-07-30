"""
Example script demonstrating how to use the NAS framework.

This script shows the complete pipeline:
1. Data loading and preprocessing 
2. Model generation with constraints
3. NASWOT scoring for model ranking
4. Bayesian optimization
5. Training and evaluation of best models
"""

import torch
import numpy as np
import random
from torch.utils.data import TensorDataset

# Import our modules
from src.models import CNNArchitecture, SEARCH_SPACE, generate_hyperparameters, check_hyperparameters
from src.optimization import (compute_naswot, extract_model_info, 
                             calculate_final_conv_output_size,
                             evaluate_models_with_naswot, rank_models_by_naswot)
from src.utils import (get_device, set_random_seeds, 
                      train_and_evaluate_model_notebook_style,
                      plot_training_history)


def create_dummy_vww_data(num_samples=1000):
    """Create dummy Visual Wake Words data for demonstration."""
    # Create random images (3, 224, 224)
    images = torch.randn(num_samples, 3, 224, 224)
    # Create random binary labels (person/no-person)
    labels = torch.randint(0, 2, (num_samples,))
    
    # Split into train/val/test
    train_size = int(0.7 * num_samples)
    val_size = int(0.15 * num_samples)
    
    train_dataset = TensorDataset(images[:train_size], labels[:train_size])
    val_dataset = TensorDataset(images[train_size:train_size+val_size], 
                               labels[train_size:train_size+val_size])
    test_dataset = TensorDataset(images[train_size+val_size:], 
                                labels[train_size+val_size:])
    
    return train_dataset, val_dataset, test_dataset


def main():
    """Main NAS pipeline demonstration."""
    print("=== Neural Architecture Search for Tiny Visual Wake Words ===")
    
    # Set seeds for reproducibility
    set_random_seeds(42)
    device = get_device()
    
    # Configuration from notebook
    MAX_MODELS = 50  # Reduced for demo
    MAX_RETRIES = 1000
    INPUT_IMAGE_SIZE = 96
    
    print("1. Creating demonstration dataset...")
    train_dataset, val_dataset, test_dataset = create_dummy_vww_data(1000)
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    print("2. Generating candidate architectures...")
    print(f"Search space: {list(SEARCH_SPACE.keys())}")
    
    # Generate models following notebook approach
    models = []
    model_hyperparams = []
    
    retry_count = 0
    while len(models) < MAX_MODELS and retry_count < MAX_RETRIES:
        # Generate random hyperparameters
        hyperparameters = generate_hyperparameters(SEARCH_SPACE)
        
        # Check constraints
        if check_hyperparameters(hyperparameters):
            try:
                # Calculate final conv output size
                final_conv_output_size = calculate_final_conv_output_size(
                    INPUT_IMAGE_SIZE, hyperparameters
                )
                hyperparameters["final_conv_output_size"] = final_conv_output_size
                
                # Create model
                model = CNNArchitecture(**hyperparameters)
                models.append(model)
                model_hyperparams.append(hyperparameters)
                
                if len(models) % 10 == 0:
                    print(f"Generated {len(models)} valid models...")
                    
            except Exception as e:
                pass  # Skip invalid models
        
        retry_count += 1
    
    print(f"Successfully generated {len(models)} models")
    
    print("3. Evaluating models with NASWOT...")
    # Create input batch for NASWOT
    dummy_input = torch.randn(32, 3, 224, 224)  # Batch of 32 images
    
    naswot_scores = []
    for i, model in enumerate(models):
        try:
            score = compute_naswot(model, dummy_input, device)
            naswot_scores.append(score)
        except Exception as e:
            naswot_scores.append(0.0)
        
        if (i + 1) % 10 == 0:
            print(f"Evaluated {i + 1}/{len(models)} models")
    
    print("4. Ranking models by NASWOT score...")
    ranked_models, ranked_scores, ranking_indices = rank_models_by_naswot(models, naswot_scores)
    
    print("Top 5 models by NASWOT score:")
    for i in range(min(5, len(ranked_models))):
        print(f"Rank {i+1}: NASWOT Score = {ranked_scores[i]:.4f}")
        model_info = extract_model_info(ranked_models[i])
        print(f"  Parameters: {model_info['Parameters']:,}")
        print(f"  FLOPs: {model_info['FLOPs']:,}")
    
    print("5. Training top model...")
    best_model = ranked_models[0]
    best_hyperparams = model_hyperparams[ranking_indices[0]]
    
    print("Best model hyperparameters:")
    for key, value in best_hyperparams.items():
        if key != "final_conv_output_size":
            print(f"  {key}: {value}")
    
    try:
        # Train with reduced epochs for demo
        train_acc_list, val_acc_list = train_and_evaluate_model_notebook_style(
            model=best_model,
            num_epochs=5,  # Reduced for demo
            batch_size=32,
            learning_rate=0.01,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            device=device
        )
        
        print("Training completed!")
        if train_acc_list:
            final_train_acc = train_acc_list[-1][1]
            final_val_acc = val_acc_list[-1][1]
            print(f"Final Training Accuracy: {final_train_acc:.2f}%")
            print(f"Final Validation Accuracy: {final_val_acc:.2f}%")
        
    except Exception as e:
        print(f"Training failed: {e}")
        print("This is expected with dummy data - use real VWW dataset for actual training")
    
    print("6. Model Analysis...")
    best_model_info = extract_model_info(best_model)
    print("Best model characteristics:")
    print(f"  Total Parameters: {best_model_info['Parameters']:,}")
    print(f"  Total FLOPs: {best_model_info['FLOPs']:,}")
    print(f"  Model Depth: {best_model_info['Depth']}")
    print(f"  Model Width: {best_model_info['Width']}")
    print(f"  Receptive Field: {best_model_info['Receptive field size']}")
    
    print("=== NAS Pipeline Complete ===")
    print("To run with real data:")
    print("1. Install pyvww: pip install pyvww")
    print("2. Download Visual Wake Words dataset")
    print("3. Update data paths in 4A.ipynb")
    print("4. Run the full notebook for complete experiments")


if __name__ == "__main__":
    main()

import torch
import numpy as np
from src.data import load_vww_data, create_data_loaders
from src.models import create_model_from_params
from src.optimization import get_search_space, create_bayesian_optimizer, extract_model_info
from src.utils import train_and_evaluate_model, get_device, set_random_seeds


def main():
    """Main function demonstrating the NAS pipeline."""
    
    # Set random seeds for reproducibility
    set_random_seeds(42)
    
    # Get device
    device = get_device()
    
    print("=== Neural Architecture Search for Tiny Visual Wake Words ===\n")
    
    # 1. Data Loading and Preparation
    print("1. Loading and preparing Visual Wake Words dataset...")
    
    # Note: Update these paths to your actual data locations
    train_root = "/path/to/train2014"
    val_root = "/path/to/val2014"
    train_ann_file = "/path/to/annotations/instances_train.json"
    val_ann_file = "/path/to/annotations/instances_val.json"
    
    try:
        train_dataset, val_dataset, test_dataset = load_vww_data(
            train_root=train_root,
            val_root=val_root,
            train_ann_file=train_ann_file,
            val_ann_file=val_ann_file
        )
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
        
    except FileNotFoundError:
        print("Data files not found. Please update the file paths.")
        print("For now, creating dummy datasets for demonstration...")
        
        # Create dummy datasets for demonstration
        from torch.utils.data import TensorDataset
        
        dummy_images = torch.randn(1000, 3, 224, 224)
        dummy_labels = torch.randint(0, 2, (1000,))
        
        train_dataset = TensorDataset(dummy_images[:700], dummy_labels[:700])
        val_dataset = TensorDataset(dummy_images[700:850], dummy_labels[700:850])
        test_dataset = TensorDataset(dummy_images[850:], dummy_labels[850:])
    
    # 2. Define Search Space
    print("\n2. Defining search space for Neural Architecture Search...")
    
    search_space = get_search_space()
    print(f"Search space parameters: {list(search_space.keys())}")
    
    # 3. Create Example Architectures
    print("\n3. Creating example architectures...")
    
    # Example parameter sets
    example_params = [
        {
            'number_of_filters4': 64, 'number_of_filters5': 128, 'number_of_filters6': 256,
            'kernel_size4': 3, 'kernel_size5': 3, 'kernel_size6': 3,
            'padding4': 1, 'padding5': 1, 'padding6': 1,
            'dropout_rate4': 0.2, 'dropout_rate5': 0.2, 'dropout_rate6': 0.2,
            'max_pooling_kernel_size4': 2, 'max_pooling_kernel_size5': 2, 'max_pooling_kernel_size6': 2,
            'max_pooling_stride4': 2, 'max_pooling_stride5': 2, 'max_pooling_stride6': 2,
            'number_of_filters10': 128, 'number_of_fc_layers': 2,
            'fc_neurons1': 64, 'fc_neurons2': 32
        },
        {
            'number_of_filters4': 32, 'number_of_filters5': 64, 'number_of_filters6': 128,
            'kernel_size4': 5, 'kernel_size5': 3, 'kernel_size6': 3,
            'padding4': 2, 'padding5': 1, 'padding6': 1,
            'dropout_rate4': 0.1, 'dropout_rate5': 0.2, 'dropout_rate6': 0.3,
            'max_pooling_kernel_size4': 2, 'max_pooling_kernel_size5': 2, 'max_pooling_kernel_size6': 2,
            'max_pooling_stride4': 2, 'max_pooling_stride5': 2, 'max_pooling_stride6': 2,
            'number_of_filters10': 64, 'number_of_fc_layers': 3,
            'fc_neurons1': 96, 'fc_neurons2': 64
        }
    ]
    
    models = []
    model_infos = []
    
    for i, params in enumerate(example_params):
        print(f"\nCreating model {i+1} with parameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        
        model = create_model_from_params(params)
        models.append(model)
        
        # Extract model information
        model_info = extract_model_info(model)
        model_infos.append(model_info)
        
        print(f"Model {i+1} info:")
        print(f"  FLOPs: {model_info['FLOPs']:,}")
        print(f"  Parameters: {model_info['Parameters']:,}")
        print(f"  Depth: {model_info['Depth']}")
        print(f"  Width: {model_info['Width']}")
    
    # 4. Demonstrate Training (on first model)
    print(f"\n4. Training example model...")
    
    example_model = models[0]
    
    try:
        train_acc, val_acc = train_and_evaluate_model(
            model=example_model,
            num_epochs=2,  # Reduced for demonstration
            batch_size=32,
            learning_rate=0.01,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            device=device
        )
        
        print(f"Final training accuracy: {train_acc[-1]:.4f}")
        print(f"Final validation accuracy: {val_acc[-1]:.4f}")
        
    except Exception as e:
        print(f"Training demonstration failed: {e}")
        print("This is expected if running without proper data setup.")
    
    # 5. Save Example Model
    print(f"\n5. Saving example model...")
    
    try:
        torch.save(example_model.state_dict(), 'models/example_model.pth')
        print("Model saved successfully!")
    except Exception as e:
        print(f"Could not save model: {e}")
    
    print("\n=== NAS Pipeline Demonstration Complete ===")
    print("\nTo run a full NAS experiment:")
    print("1. Set up the Visual Wake Words dataset")
    print("2. Configure the search space and optimization parameters")
    print("3. Run Bayesian optimization for architecture search")
    print("4. Train and evaluate the best found architectures")
    print("5. Compare results and select the final model")


if __name__ == "__main__":
    main()
