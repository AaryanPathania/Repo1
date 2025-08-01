#!/usr/bin/env python3
"""
Alzheimer's Disease Prediction Demo
===================================

A simplified and runnable version of the AD prediction project that can work with synthetic data.
This script demonstrates both 2D AlexNet and 3D CNN approaches for AD classification.

Usage:
    python ad_prediction_demo.py --mode alexnet --epochs 10 --batch_size 8
    python ad_prediction_demo.py --mode autoencoder --epochs 5 --batch_size 4
"""

import argparse
import os
import sys
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import demo data generator
from demo_data_generator import SyntheticBrainGenerator

# Configure logging
logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', 
    level=logging.INFO
)

class SimpleAlexNet(nn.Module):
    """Simplified AlexNet for brain MRI classification"""
    
    def __init__(self, num_classes=2):
        super(SimpleAlexNet, self).__init__()
        
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Second conv block
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Third conv block
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Fourth conv block
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Fifth conv block
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class Simple3DCNN(nn.Module):
    """Simplified 3D CNN for brain MRI classification"""
    
    def __init__(self, num_classes=2):
        super(Simple3DCNN, self).__init__()
        
        self.features = nn.Sequential(
            # First 3D conv block
            nn.Conv3d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),
            
            # Second 3D conv block
            nn.Conv3d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),
            
            # Third 3D conv block
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),
        )
        
        # Calculate the size after conv layers
        # Input: (121, 145, 121) -> After convs: approximately (4, 5, 4)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 5 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class BrainMRIDataset(torch.utils.data.Dataset):
    """Dataset class for brain MRI data"""
    
    def __init__(self, data_file, data_dir="./demo_data", mode="2d", transform=None):
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform
        
        # Read data file
        with open(data_file, 'r') as f:
            lines = f.readlines()
        
        self.samples = []
        for line in lines:
            filename, label = line.strip().split()
            self.samples.append((filename, label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        filename, label = self.samples[idx]
        
        # Load brain image
        import nibabel as nib
        filepath = os.path.join(self.data_dir, filename)
        img = nib.load(filepath)
        data = img.get_fdata()
        
        if self.mode == "2d":
            # Extract 2D slices and create RGB image (as in original paper)
            axial_slice = data[:, :, 78]  # Key position from paper
            coronal_slice = data[:, 79, :]  # Key position from paper  
            sagittal_slice = data[57, :, :]  # Key position from paper
            
            # Resize slices to same size
            from skimage.transform import resize
            target_size = (224, 224)  # AlexNet input size
            
            axial_slice = resize(axial_slice, target_size)
            coronal_slice = resize(coronal_slice, target_size)
            sagittal_slice = resize(sagittal_slice, target_size)
            
            # Create RGB image
            rgb_image = np.stack([axial_slice, coronal_slice, sagittal_slice], axis=0)
            
            # Normalize
            rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min() + 1e-8)
            
            # Apply transforms
            if self.transform:
                rgb_image = self.transform(rgb_image)
            else:
                rgb_image = torch.FloatTensor(rgb_image)
            
            sample = rgb_image
            
        else:  # 3D mode
            # Normalize 3D data
            data = (data - data.min()) / (data.max() - data.min() + 1e-8)
            
            # Resize to smaller size for memory efficiency
            from skimage.transform import resize
            data = resize(data, (64, 72, 64))  # Smaller than original for demo
            
            sample = torch.FloatTensor(data).unsqueeze(0)  # Add channel dimension
        
        # Convert label
        label_idx = 0 if label == "Normal" else 1
        
        return sample, label_idx

def train_model(model, train_loader, val_loader, num_epochs, device, learning_rate=1e-4):
    """Train the model"""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_accuracies = []
    
    model.to(device)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_loss = 0.0
        
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (data, target) in enumerate(train_progress):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            train_progress.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Validation phase
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        val_accuracy = 100 * correct / total
        val_accuracies.append(val_accuracy)
        
        logging.info(f'Epoch {epoch+1}: Loss: {avg_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')
    
    return train_losses, val_accuracies

def test_model(model, test_loader, device):
    """Test the model and return accuracy"""
    model.eval()
    correct = 0
    total = 0
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(target.cpu().numpy())
    
    accuracy = 100 * correct / total
    return accuracy, predictions, true_labels

def plot_results(train_losses, val_accuracies, save_path="training_results.png"):
    """Plot training results"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(train_losses)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    ax2.plot(val_accuracies)
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='AD Prediction Demo')
    parser.add_argument('--mode', choices=['alexnet', '3dcnn', 'both'], default='alexnet',
                       help='Model to use: alexnet, 3dcnn, or both')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--generate_data', action='store_true',
                       help='Generate new synthetic data')
    parser.add_argument('--n_samples', type=int, default=30,
                       help='Number of samples per class to generate')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    
    # Generate data if requested or if data files don't exist
    data_files = ['demo_train_2classes.txt', 'demo_validation_2classes.txt', 'demo_test_2classes.txt']
    if args.generate_data or not all(os.path.exists(f) for f in data_files):
        logging.info('Generating synthetic brain data...')
        generator = SyntheticBrainGenerator()
        generator.generate_dataset(n_normal=args.n_samples, n_ad=args.n_samples)
        logging.info('Data generation complete!')
    
    # Run experiments based on mode
    if args.mode in ['alexnet', 'both']:
        logging.info('Starting AlexNet (2D) experiment...')
        
        # Create datasets
        train_dataset = BrainMRIDataset('demo_train_2classes.txt', mode='2d')
        val_dataset = BrainMRIDataset('demo_validation_2classes.txt', mode='2d')
        test_dataset = BrainMRIDataset('demo_test_2classes.txt', mode='2d')
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        # Create model
        model = SimpleAlexNet(num_classes=2)
        
        # Train model
        train_losses, val_accuracies = train_model(
            model, train_loader, val_loader, args.epochs, device, args.learning_rate
        )
        
        # Test model
        test_accuracy, predictions, true_labels = test_model(model, test_loader, device)
        logging.info(f'AlexNet Test Accuracy: {test_accuracy:.2f}%')
        
        # Plot results
        plot_results(train_losses, val_accuracies, 'alexnet_results.png')
        
        # Save model
        torch.save(model.state_dict(), 'alexnet_model.pth')
    
    if args.mode in ['3dcnn', 'both']:
        logging.info('Starting 3D CNN experiment...')
        
        # Create datasets  
        train_dataset = BrainMRIDataset('demo_train_2classes.txt', mode='3d')
        val_dataset = BrainMRIDataset('demo_validation_2classes.txt', mode='3d')
        test_dataset = BrainMRIDataset('demo_test_2classes.txt', mode='3d')
        
        # Create data loaders (smaller batch size for 3D)
        batch_size_3d = max(1, args.batch_size // 2)
        train_loader = DataLoader(train_dataset, batch_size=batch_size_3d, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size_3d, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size_3d, shuffle=False)
        
        # Create model
        model = Simple3DCNN(num_classes=2)
        
        # Train model
        train_losses, val_accuracies = train_model(
            model, train_loader, val_loader, args.epochs, device, args.learning_rate
        )
        
        # Test model
        test_accuracy, predictions, true_labels = test_model(model, test_loader, device)
        logging.info(f'3D CNN Test Accuracy: {test_accuracy:.2f}%')
        
        # Plot results
        plot_results(train_losses, val_accuracies, '3dcnn_results.png')
        
        # Save model
        torch.save(model.state_dict(), '3dcnn_model.pth')
    
    logging.info('Experiment completed successfully!')

if __name__ == '__main__':
    main()