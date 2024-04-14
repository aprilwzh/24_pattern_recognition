import utils
import numpy as np
import matplotlib.pyplot as plt
import time
from MLP import MLP, train, evaluate_model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torchvision.transforms as transforms
import multiprocessing
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    folder_name = 'Fashion-MNIST/'
    file_name = 'gt-test.tsv'

    # Load test data
    start = time.time()
    test_file_names, _ = utils.read_file(folder_name, file_name, train=False, has_labels=False)  # Set has_labels to False
    test_samples = utils.load_files(folder_name, test_file_names)
    end = time.time()

    print(f'time elapsed: {round(end - start, 2)}')

    # Load train data
    file_name = 'gt-train.tsv'
    start = time.time()
    train_file_names, train_labels = utils.read_file(folder_name, file_name, train=True)
    train_samples = utils.load_files(folder_name, train_file_names)
    end = time.time()

    print(f'time elapsed: {round(end - start, 2)}')

    # Split the training data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(train_samples, train_labels, test_size=0.2, random_state=0)

    # Define the hyperparameters
    input_size = 28 * 28  # Input size assuming Fashion MNIST images
    hidden_size = 128
    output_size = 10  # Output size for 10 classes in Fashion MNIST
    learning_rate = 0.001
    num_epochs = 10

    # Convert numpy arrays to tensors
    train_data_tensor = torch.from_numpy(X_train).float()
    train_labels_tensor = torch.from_numpy(y_train.reshape(-1)).long()  # Reshape and convert to tensor
    val_data_tensor = torch.from_numpy(X_val).float()
    val_labels_tensor = torch.from_numpy(y_val.reshape(-1)).long()  # Reshape and convert to tensor

    # Create datasets
    train_dataset = TensorDataset(train_data_tensor, train_labels_tensor)
    val_dataset = TensorDataset(val_data_tensor, val_labels_tensor)

    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=multiprocessing.cpu_count())
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=multiprocessing.cpu_count())

    # Initialize the MLP model
    model = MLP(input_size, hidden_size, output_size)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_losses, valid_losses = train(model, criterion, optimizer, train_loader, val_loader, num_epochs)

    # Plot loss curves
    plt.plot(train_losses, label='Train Loss')
    plt.plot(valid_losses, label='Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Convert test samples to tensor
    test_data_tensor = torch.from_numpy(test_samples).float()

    # Create a Dataset from test data tensor
    test_dataset = TensorDataset(test_data_tensor)

    # Create a DataLoader from the test Dataset
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=multiprocessing.cpu_count())

    # Evaluate the model on the test set
    test_accuracy = evaluate_model(model, test_loader)
    print(f'Test Accuracy: {test_accuracy:.4f}')
