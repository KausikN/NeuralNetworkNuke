"""
Model Utils
"""

# Imports
import pickle
from keras.models import load_model
import torchvision.models as models

# Main Vars


# Main Functions
def Model_LoadPickle(path, **params):
    '''
    Load a pickled model file
    '''
    with open(path, "rb") as f: return pickle.load(f)

def Model_LoadKeras(path, **params):
    '''
    Load a Keras model
    '''
    return load_model(path)

def Model_LoadModel(path, **params):
    '''
    Load a model
    '''
    ext = path.split(".")[-1]
    # Pickle
    if ext in ["p", "pickle"]:
        return Model_LoadPickle(path, **params)
    # Keras
    elif ext in ["h5", "hdf5"]:
        return Model_LoadKeras(path, **params)
    else:
        model = models.alexnet(pretrained=True)
        model.eval()
        return model
    # None
    return None