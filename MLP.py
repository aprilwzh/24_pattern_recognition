import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.metrics import accuracy_score

# Load the training and test datasets
train_data = pd.read_csv('MNIST-full\\MNIST-full\\gt-train.tsv', sep='\t', header=None)
test_data = pd.read_csv('MNIST-full\\MNIST-full\\gt-test.tsv', sep='\t', header=None)

# Extract features and labels
X_train = train_data.drop(columns=[0]).values
y_train = train_data[0].values

X_test = test_data.drop(columns=[0]).values
y_test = test_data[0].values

# Create a group column with two groups
train_data['group'] = np.repeat([0, 1], len(train_data) // 2)

# Define the MLP classifier
mlp = MLPClassifier(max_iter=10000)  # Increase the maximum number of iterations

# Define the hyperparameter grid
param_grid = {
    'hidden_layer_sizes': np.arange(10, 257, 10),
    'learning_rate_init': np.logspace(-4, -1, 4)
}

# Perform grid search with group cross-validation with 2 groups
group_cv = GroupKFold(n_splits=2)
grid_search = GridSearchCV(mlp, param_grid, cv=group_cv, n_jobs=-1)
grid_search.fit(X_train, y_train, groups=train_data['group'])

# Get the best parameters and best accuracy
best_params = grid_search.best_params_
best_accuracy = grid_search.best_score_

# Train the MLP classifier with the best parameters
best_mlp = MLPClassifier(hidden_layer_sizes=best_params['hidden_layer_sizes'],
                         learning_rate_init=best_params['learning_rate_init'],
                         max_iter=10000)  # Increase max_iter for better convergence
best_mlp.fit(X_train, y_train)

# Predict on the test set
y_pred = best_mlp.predict(X_test)

# Calculate accuracy on the test set
accuracy = accuracy_score(y_test, y_pred)

print("Best parameters:", best_params)
print("Best accuracy:", best_accuracy)
print("Accuracy on test set:", accuracy)
