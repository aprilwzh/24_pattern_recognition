import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import multiprocessing
import utils
import time

# Check number of cores available for computing SVM
n_cores = multiprocessing.cpu_count()
print(n_cores)

# Folder to dataset
folder_name = 'MNIST-full/'

# Read train data files
start = time.time()
train_data_files, y_train = utils.read_file(folder_name=folder_name, file_name='gt-train.tsv', train=True)
train_data = utils.load_files(folder_name=folder_name, file_names=train_data_files)
print('train data read')
print(f'Time elapsed: {round(time.time() - start, 2)}')

# Read test data files
start = time.time()
test_data_files, y_test = utils.read_file(folder_name=folder_name, file_name='gt-test.tsv', train=False)
test_data = utils.load_files(folder_name=folder_name, file_names=test_data_files)
print('test data read')
print(f'Time elapsed: {round(time.time() - start, 2)}')

# Extract labels and features
X_train = np.reshape(train_data, (len(train_data), 784), order='C')
X_test = np.reshape(test_data, (len(test_data), 784), order='C')
print('data flattened')

# Normalize data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
print('data scaled')

# Define parameter grid for grid search
# Loop over both Linear and RBF kernels
for kernel in ['linear', 'rbf']:
    # C values to test
    param_grid = {'C': [0.1, 1, 10, 100]}

    # Perform grid search with cross-validation
    svm = SVC(kernel=kernel, verbose=False, max_iter=500)  # max_iter reduces runtime, raises ConvergenceWarning...
    grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, verbose=3,
                               cv=3,  # 3 fold cross validation
                               n_jobs=n_cores-2)  # use all but 2 available cores to keep computer functional...
    grid_search.fit(X_train, y_train) # Fun grid search to find optimal parameters

    # Get the best model
    best_model = grid_search.best_estimator_
    print(f"kernel: {kernel}, C: {grid_search.best_params_['C']}")

    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
