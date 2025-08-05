"""
Neural Architecture Search Model for Tiny Visual Wake Words
Extracted from 4A.ipynb notebook
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from tqdm import tqdm


class BalancedVisualWakeWordsDataset(Dataset):
    """Custom Dataset class for balanced Visual Wake Words dataset"""
    
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image, label = self.dataset[self.indices[idx]]
        if self.transform:
            image = self.transform(image)
        return image, label


class CustomDNN(nn.Module):
    """Custom Deep Neural Network for Visual Wake Words Classification"""
    
    def __init__(self, hyperparams):
        super(CustomDNN, self).__init__()

        self.blocks = nn.Sequential()
        for i in range(4, 7):
            in_channels = hyperparams[f"number_of_filters{i - 1}"] if i > 4 else 3
            out_channels = hyperparams[f"number_of_filters{i}"]
            kernel_size = hyperparams[f"kernel_size{i}"]
            padding = hyperparams[f"padding{i}"]
            dropout_rate = hyperparams[f"dropout_rate{i}"]
            max_pooling_kernel_size = hyperparams[f"max_pooling_kernel_size{i}"]
            max_pooling_stride = hyperparams[f"max_pooling_stride{i}"]

            self.blocks.add_module(f"block{i}", nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.MaxPool2d(max_pooling_kernel_size, stride=max_pooling_stride)
            ))

        self.fc = nn.Sequential()
        in_features = hyperparams["final_conv_output_size"]
        for i in range(hyperparams["number_of_fc_layers"]):
            out_features = hyperparams["number_of_neurons_per_fc_layer"]
            self.fc.add_module(f"fc{i}", nn.Linear(in_features, out_features))
            self.fc.add_module(f"relu{i}", nn.ReLU(inplace=True))
            in_features = out_features

        self.fc.add_module("dropout_fc", nn.Dropout(hyperparams["dropout_rate_fc"]))
        self.fc.add_module("output", nn.Linear(in_features, 2))
        self.fc.add_module("softmax", nn.Softmax(dim=1))

    def forward(self, x):
        x = self.blocks(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def train_and_evaluate_model(model, num_epochs, batch_size, learning_rate, 
                           train_dataset, val_dataset, test_dataset, device):
    """Complete training and evaluation pipeline"""
    
    def train(model, dataloader, criterion, optimizer, device, grad_clip=1.0):
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(dataloader)
        accuracy = 100 * correct / total
        return epoch_loss, accuracy

    def validate(model, dataloader, criterion, device):
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(dataloader, desc="Validation"):
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

    # Setup training components
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.1)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    best_val_loss = float('inf')
    patience = 10
    counter = 0
    train_acc_list = []
    val_acc_list = []

    # Training loop
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
