#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


# Load training and testing data
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

# Extract labels and features
X_train = train_data.iloc[:, 1:].values  # Features (pixel values)
y_train = train_data.iloc[:, 0].values  # Labels

X_test = test_data.iloc[:, 1:].values  # Features (pixel values)
y_test = test_data.iloc[:, 0].values  # Labels

# Define parameter grid for grid search
param_grid = {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf']}

# Perform grid search with cross-validation
svm = SVC()
grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_

# Evaluate on test set
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# In[6]:


# pip install tensorflow


# In[7]:


# b Multilayer Perceptron (MLP)
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load training and testing data
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

# Extract labels and features
X_train = train_data.iloc[:, 1:].values  # Features (pixel values)
y_train = train_data.iloc[:, 0].values  # Labels

X_test = test_data.iloc[:, 1:].values  # Features (pixel values)
y_test = test_data.iloc[:, 0].values  # Labels

# Encode labels
encoder = LabelEncoder()
encoder.fit(y_train)
encoded_y_train = encoder.transform(y_train)
encoded_y_test = encoder.transform(y_test)


# Define create_model function for KerasClassifier
def create_model(hidden_size=32, learning_rate=0.01):
    model = Sequential()
    model.add(Dense(hidden_size, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# Create and compile the model
model = create_model()

# Train the model
model.fit(X_train, encoded_y_train, epochs=1000, batch_size=64, validation_split=0.3)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, encoded_y_test, verbose=False)
print('Test accuracy:', test_acc)

# In[9]:


##   Convolutional Neural Network (CNN)
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Load training and testing data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Extract labels and features
X_train = train_data.iloc[:, 1:].values.reshape(-1, 28, 28, 1)  # Features (pixel values)
y_train = train_data.iloc[:, 0].values  # Labels

X_test = test_data.iloc[:, 1:].values.reshape(-1, 28, 28, 1)  # Features (pixel values)
y_test = test_data.iloc[:, 0].values  # Labels

# Normalize pixel values
X_train, X_test = X_train / 255.0, X_test / 255.0

# Create CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=5, validation_split=0.2)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)

# In[2]:


## C should be final one, but i have some problems
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
# from tensorflow.keras import KerasClassifier
from scikeras.wrappers import KerasClassifier, KerasRegressor

# Load training and testing data
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

# Extract labels and features
X_train = train_data.iloc[:, 1:].values.reshape(-1, 28, 28, 1)  # Features (pixel values)
y_train = train_data.iloc[:, 0].values  # Labels

X_test = test_data.iloc[:, 1:].values.reshape(-1, 28, 28, 1)  # Features (pixel values)
y_test = test_data.iloc[:, 0].values  # Labels

# Normalize pixel values
X_train, X_test = X_train / 255.0, X_test / 255.0


# Define create_model function for Keras model
def create_model(kernel_size=3, num_conv_layers=2, learning_rate=0.01):
    model = Sequential()
    model.add(Conv2D(32, kernel_size, activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    for _ in range(num_conv_layers - 1):
        model.add(Conv2D(64, kernel_size, activation='relu'))
        model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# Create Keras model
model = create_model()

# Define parameter grid for grid search
param_grid = {
    'kernel_size': [3, 5, 7],
    'num_conv_layers': [1, 2, 3],
    'learning_rate': [0.0001, 0.001, 0.01, 0.1]
}

# Wrap Keras model in a scikit-learn estimator
keras_estimator = KerasClassifier(build_fn=create_model, epochs=5, batch_size=32, verbose=0, kernel_size=3,
                                  learning_rate=0.0001, num_conv_layers=2)

# Perform grid search with cross-validation
grid_search = GridSearchCV(estimator=keras_estimator, param_grid=param_grid, cv=3)
grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_

# Evaluate on test set
test_loss, test_acc = best_model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)
## I cant load the module tensorflow.keras.wrappers.


# In[4]:


from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from scikeras.wrappers import KerasClassifier, KerasRegressor


# Define the CNN model
def create_model(kernel_size=3, num_conv_layers=2, learning_rate=0.01):
    model = Sequential()
    model.add(Conv2D(32, kernel_size, activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    for _ in range(num_conv_layers - 1):
        model.add(Conv2D(64, kernel_size, activation='relu'))
        model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# Define parameter grid for grid search
param_grid = {
    'kernel_size': [3, 5, 7],
    'num_conv_layers': [1, 2, 3],
    'learning_rate': [0.0001, 0.001, 0.01, 0.1]
}

# Create KerasRegressor
model = KerasClassifier(build_fn=create_model, epochs=5, batch_size=32, verbose=0, kernel_size=3, learning_rate=0.0001,
                        num_conv_layers=2)

# Perform grid search with cross-validation
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
grid_result = grid_search.fit(X_train, y_train)

# Get the best parameters and best score
best_params = grid_result.best_params_
best_score = grid_result.best_score_

# Print the best parameters and best score
print("Best Parameters:", best_params)
print("Best Score:", best_score)

# In[ ]:

# !pip install tensorflow==2.12.0
# !pip install keras


# In[ ]:
# CNN
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

# Load training and testing data
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

# Extract labels and features
X_train = train_data.iloc[:, 1:].values.reshape(-1, 28, 28, 1)  # Features (pixel values)
y_train = train_data.iloc[:, 0].values  # Labels

X_test = test_data.iloc[:, 1:].values.reshape(-1, 28, 28, 1)  # Features (pixel values)
y_test = test_data.iloc[:, 0].values  # Labels

# Normalize pixel values
X_train, X_test = X_train / 255.0, X_test / 255.0


# Define create_model function for Keras model
def create_model(kernel_size=3, num_conv_layers=2, learning_rate=0.01):
    model = Sequential()
    model.add(Conv2D(32, kernel_size, activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    for _ in range(num_conv_layers - 1):
        model.add(Conv2D(64, kernel_size, activation='relu'))
        model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# Create Keras model
model = KerasClassifier(build_fn=create_model, epochs=5, batch_size=32, verbose=0)

# Define parameter grid for grid search
param_grid = {
    'kernel_size': [3, 5, 7],
    'num_conv_layers': [1, 2, 3],
    'learning_rate': [0.0001, 0.001, 0.01, 0.1]
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_

# Evaluate on test set
test_acc = best_model.score(X_test, y_test)
print('Test accuracy:', test_acc)
