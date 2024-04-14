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

if __name__ == '__main__':
    folder_name = 'MNIST-full/'
    file_name = 'gt-test.tsv'

    # Load test data
    start = time.time()
    test_file_names, test_labels = utils.read_file(folder_name, file_name, train=False)
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

    # Define the hyperparameters
    input_size = 28 * 28  # Input size assuming MNIST images (= 784)
    hidden_size = 128
    output_size = 10  # Output size for 10 classes in MNIST
    learning_rate = 0.001
    num_epochs = 10

    # Convertir numpy arrays en tensors
    train_data_tensor = torch.from_numpy(train_samples).float()
    train_labels_tensor = torch.from_numpy(train_labels.reshape(-1)).long()  # Redimensionner et convertir en tensor
    test_data_tensor = torch.from_numpy(test_samples).float()
    test_labels_tensor = torch.from_numpy(test_labels.reshape(-1)).long()  # Redimensionner et convertir en tensor
    
    train_dataset = TensorDataset(train_data_tensor, train_labels_tensor)
    test_dataset = TensorDataset(test_data_tensor, test_labels_tensor)

    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])


    train_dataset = TensorDataset(train_data_tensor, train_labels_tensor)
    test_dataset = TensorDataset(test_data_tensor, test_labels_tensor)

    # Create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=multiprocessing.cpu_count())
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=multiprocessing.cpu_count())

    # Initialize the MLP model
    model = MLP(input_size, hidden_size, output_size)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_losses, valid_losses = train(model, criterion, optimizer, train_loader, train_loader, num_epochs)

    # Plot loss curves
    plt.plot(train_losses, label='Train Loss')
    plt.plot(valid_losses, label='Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Evaluate the model on test set
    test_accuracy = evaluate_model(model, test_loader)
    print(f'Test Accuracy: {test_accuracy:.4f}')
