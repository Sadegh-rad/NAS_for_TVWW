"""
Utility Functions Module

This module contains various utility functions for training, evaluation,
visualization, and other helper functions used throughout the project.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def train_model(model, train_loader, criterion, optimizer, device):
    """
    Train model for one epoch.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        criterion: Loss criterion
        optimizer: Optimizer
        device: Training device (CPU/GPU)
        
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    for inputs, labels in tqdm(train_loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()
    
    avg_loss = running_loss / len(train_loader)
    accuracy = correct_predictions / total_samples
    
    return avg_loss, accuracy


def evaluate_model(model, data_loader, criterion, device):
    """
    Evaluate model on given dataset.
    
    Args:
        model: PyTorch model
        data_loader: Data loader for evaluation
        criterion: Loss criterion
        device: Evaluation device (CPU/GPU)
        
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
    
    avg_loss = running_loss / len(data_loader)
    accuracy = correct_predictions / total_samples
    
    return avg_loss, accuracy


def train_and_evaluate_model(model, num_epochs, batch_size, learning_rate,
                           train_dataset, val_dataset, test_dataset, device):
    """
    Complete training and evaluation pipeline for a model.
    
    Args:
        model: PyTorch model to train
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        device: Training device
        
    Returns:
        tuple: (train_accuracies, val_accuracies)
    """
    # Create data loaders
    from torch.utils.data import DataLoader
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Setup training components
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Move model to device
    model = model.to(device)
    
    # Training history
    train_accuracies = []
    val_accuracies = []
    train_losses = []
    val_losses = []
    
    print(f"Training model for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Training
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
        train_accuracies.append(train_acc)
        train_losses.append(train_loss)
        
        # Validation
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        val_accuracies.append(val_acc)
        val_losses.append(val_loss)
        
        # Update learning rate
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # Final evaluation on test set
    _, test_acc = evaluate_model(model, test_loader, criterion, device)
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")
    
    return train_accuracies, val_accuracies


def train_and_evaluate_model_notebook_style(model, num_epochs, batch_size, learning_rate,
                                           train_dataset, val_dataset, test_dataset, device):
    """
    Training function matching the exact implementation from 4A notebook.
    
    Args:
        model: PyTorch model to train
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        device: Training device
        
    Returns:
        tuple: (train_accuracies, val_accuracies)
    """
    from torch.utils.data import DataLoader
    
    def train(model, dataloader, criterion, optimizer, device):
        """Training function from notebook."""
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(dataloader, desc="Training"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(dataloader)
        accuracy = 100 * correct / total
        return epoch_loss, accuracy

    def validate(model, dataloader, criterion, device):
        """Validation function from notebook."""
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(dataloader, desc="Validating"):
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(dataloader)
        accuracy = 100 * correct / total
        return epoch_loss, accuracy

    def test(model, dataloader, device):
        """Test function from notebook."""
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(dataloader, desc="Testing"):
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        accuracy = 100 * correct / total
        return accuracy

    # Setup as in notebook
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.1)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    best_val_loss = float('inf')
    patience = 10
    counter = 0
    train_acc_list = []
    val_acc_list = []

    model = model.to(device)

    # Training and validation loop (from notebook)
    for epoch in range(num_epochs):
        train_loss, train_acc = train(model.half(), train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        print(f"Epoch {epoch + 1}/{num_epochs}:")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")
        
        val_acc_list.append((epoch+1, val_acc))
        train_acc_list.append((epoch+1, train_acc))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_name = type(model).__name__
            torch.save(model.state_dict(), f'{model_name}.pth')
            counter = 0
        else:
            counter += 1

            if counter >= patience:
                print("Early stopping triggered. Stopping training.")
                break

    # Load best model and test
    model.load_state_dict(torch.load(f'{model_name}.pth'))
    test_acc = test(model, test_loader, device)
    print(f"Test Accuracy: {test_acc:.2f}%")

    return train_acc_list, val_acc_list


def plot_training_history(train_accuracies, val_accuracies, title="Training History"):
    """
    Plot training and validation accuracy curves.
    
    Args:
        train_accuracies: List of training accuracies
        val_accuracies: List of validation accuracies
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(train_accuracies) + 1)
    
    plt.plot(epochs, train_accuracies, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy')
    
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()


def unnormalize_image(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Unnormalize image tensor for visualization.
    
    Args:
        img_tensor: Normalized image tensor
        mean: Normalization mean values
        std: Normalization std values
        
    Returns:
        numpy.ndarray: Unnormalized image
    """
    img_np = img_tensor.numpy().transpose((1, 2, 0)).astype(np.float32)
    img_np = img_np * np.array(std) + np.array(mean)
    img_np = np.clip(img_np, 0, 1)
    return img_np


def plot_random_images(dataset, title, num_images=10, mean=[0.485, 0.456, 0.406], 
                      std=[0.229, 0.224, 0.225]):
    """
    Plot random images from dataset with labels.
    
    Args:
        dataset: Dataset to sample from
        title: Plot title
        num_images: Number of images to display
        mean: Normalization mean values
        std: Normalization std values
    """
    import random
    
    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
    fig.suptitle(title)

    for ax in axes:
        idx = random.randint(0, len(dataset) - 1)
        img, label = dataset[idx]
        img_np = unnormalize_image(img, mean, std)

        ax.imshow(img_np)
        ax.set_title(f"Label: {label}")
        ax.axis("off")

    plt.show()


def save_model(model, filepath):
    """
    Save model state dictionary.
    
    Args:
        model: PyTorch model
        filepath: Path to save model
    """
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")


def load_model(model, filepath, device):
    """
    Load model state dictionary.
    
    Args:
        model: PyTorch model (architecture)
        filepath: Path to load model from
        device: Device to load model on
        
    Returns:
        model: Loaded model
    """
    model.load_state_dict(torch.load(filepath, map_location=device))
    model.to(device)
    print(f"Model loaded from {filepath}")
    return model


def get_device():
    """
    Get the best available device (CUDA if available, else CPU).
    
    Returns:
        torch.device: Available device
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    return device


def set_random_seeds(seed=42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    
    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_model_summary(model, input_size=(1, 3, 224, 224)):
    """
    Print detailed model summary.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size
    """
    try:
        from torchinfo import summary
        summary(model, input_size=input_size)
    except ImportError:
        print("torchinfo not available. Install with: pip install torchinfo")
        print(f"Model: {model}")


def calculate_model_size(model):
    """
    Calculate model size in MB.
    
    Args:
        model: PyTorch model
        
    Returns:
        float: Model size in MB
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.numel() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb
