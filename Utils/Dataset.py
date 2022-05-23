"""
Dataset Functions
"""

# Imports
import torch
from keras.datasets import mnist
import numpy as np

# Main Vars
DATASET_IMAGE_SHAPE = (28, 28)

# Main Functions
def Dataset_Load_MNIST(verbose=False):
    '''
    Load MNIST dataset
    '''
    # Load Data
    if verbose: print("Loading MNIST Dataset...")
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    # Preprocess Data
    # Reshape Data
    if verbose: print("Preprocessing Dataset...")
    X_train = X_train.reshape(X_train.shape[0], 1, DATASET_IMAGE_SHAPE[0], DATASET_IMAGE_SHAPE[1])
    X_test = X_test.reshape(X_test.shape[0], 1, DATASET_IMAGE_SHAPE[0], DATASET_IMAGE_SHAPE[1])
    # Normalize Data
    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")
    X_train /= 255.0
    X_test /= 255.0
    
    X_train = np.array(X_train).reshape(60000, DATASET_IMAGE_SHAPE[0]*DATASET_IMAGE_SHAPE[1])

    if verbose: print("X_train:", X_train.shape)
    if verbose: print("Y_train:", Y_train.shape)
    if verbose: print("X_test:", X_test.shape)
    if verbose: print("Y_test:", Y_test.shape)

    X_train = torch.from_numpy(X_train)
    X_test = torch.from_numpy(X_test)
    Y_train = torch.from_numpy(Y_train.astype(int))
    Y_test = torch.from_numpy(Y_test.astype(int))

    DATASET = {
        "X_train": X_train,
        "Y_train": Y_train,
        "X_test": X_test,
        "Y_test": Y_test
    }

    return DATASET