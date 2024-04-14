#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#c,CNN
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class CNN(nn.Module):
    def __init__(self, num_classes=10, num_conv_layers=2, kernel_size=3):
        super(CNN, self).__init__()
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(nn.Conv2d(3, 32, kernel_size, padding=1))
        self.conv_layers.append(nn.ReLU())
        self.conv_layers.append(nn.MaxPool2d(2))
        
        for _ in range(num_conv_layers - 1):
            self.conv_layers.append(nn.Conv2d(32, 64, kernel_size, padding=1))
            self.conv_layers.append(nn.ReLU())
            self.conv_layers.append(nn.MaxPool2d(2))

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64 * 7 * 7, num_classes)  # Assuming input image size is 28x28 after pooling

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

# Function to train the CNN
def train_cnn(model, train_loader, val_loader, criterion, optimizer, num_epochs=5):
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_accuracy = correct_train / total_train
        train_loss = running_train_loss / len(train_loader)
        train_loss_history.append(train_loss)
        train_acc_history.append(train_accuracy)

        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_accuracy = correct_val / total_val
        val_loss = running_val_loss / len(val_loader)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_accuracy)

        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')

    # Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs+1), train_loss_history, label='Training Loss')
    plt.plot(range(1, num_epochs+1), val_loss_history, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot accuracy curves
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs+1), train_acc_history, label='Training Accuracy')
    plt.plot(range(1, num_epochs+1), val_acc_history, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

# Main function to train and evaluate the CNN
def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    num_epochs = 5
    batch_size = 64
    learning_rate = 0.001
    num_conv_layers = 2
    kernel_size = 3

    # Data preprocessing and loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.ImageFolder(root='/Users/april/Downloads/MNIST-full', transform=transform) #data_path
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = datasets.ImageFolder(root='/Users/april/Downloads/MNIST-full', transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model, Loss, Optimizer
    model = CNN(num_conv_layers=num_conv_layers, kernel_size=kernel_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_cnn(model, train_loader, val_loader, criterion, optimizer, num_epochs=num_epochs)

if __name__ == "__main__":
    main()

