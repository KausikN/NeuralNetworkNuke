"""
Model Utils
"""

# Imports
import pickle

# Main Vars


# Main Functions
def Model_LoadPickle(path):
    '''
    Load a pickled model file
    '''
    with open(path, "rb") as f: return pickle.load(f)