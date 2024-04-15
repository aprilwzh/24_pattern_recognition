import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import multiprocessing
import utils
import csv

# Find number of cores available for processing
n_cores = multiprocessing.cpu_count()
print(n_cores)

# Folder name for Competition dataset
folder_name = 'Fashion-MNIST/'

# Load train data
train_data_files, y_train = utils.read_file(folder_name=folder_name, file_name='gt-train.tsv', train=True)
train_data = utils.load_files(folder_name=folder_name, file_names=train_data_files)
print('train data read')

# Load test data
test_data_files, _ = utils.read_file(folder_name=folder_name, file_name='gt-test.tsv', train=False, has_labels=False)
test_data = utils.load_files(folder_name=folder_name, file_names=test_data_files)
print('test data read')

# Extract labels and features
X_train = np.reshape(train_data, (len(train_data), 784), order='C')
X_test = np.reshape(test_data, (len(test_data), 784), order='C')
print('data flattened')

# Normalize data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
print('data scaled')

# Split Train Set to find initial best parameters
X_train2, X_val, y_train2, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

# Define C parameters
param_grid = {'C': [0.1, 1, 10, 100]}

# Perform grid search with cross-validation
svm = SVC(kernel='rbf', verbose=False, max_iter=500)
grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=3, verbose=3, n_jobs=n_cores-2)
grid_search.fit(X_train2, y_train2)

# Get the best model
best_model = grid_search.best_estimator_
print(f"kernel: rbf, C: {grid_search.best_params_['C']}")

# Evaluate on test set
y_pred = best_model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print("Accuracy:", accuracy)

svm2 = SVC(kernel='rbf', verbose=False, max_iter=500, C=grid_search.best_params_['C'])  # Extract best model

svm2.fit(X_train, y_train)  # Fit FULL dataset instead of subset like above

y_test_pred = svm2.predict(X_test)  # Predict test set

print('starting here')
print(test_data_files[:10])
print(y_test_pred[:10])

# Write to gt-test.tsv our filenames and predictions
file_name = 'out/gt-test.tsv'

with open(file_name, 'w', newline='') as tsvfile:
    writer = csv.writer(tsvfile, delimiter='\t', lineterminator='\n')
    for name, label in zip(test_data_files, y_test_pred):
        writer.writerow([name, label])

