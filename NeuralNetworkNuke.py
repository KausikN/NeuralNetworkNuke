"""
Neural Network Nuke
"""

# Imports
from Utils.Dataset import *
from Utils.Model import *
from Models.ModelTemplates import *

from Library.Attacks.FGSM import *

# Main Vars
MODELS = {
    "mnist": {
        "model_path": "Models/MNIST/bestmodel_mnist.p",#"Models/MNIST/bestmodel_mnist.hdf5",
        "dataset_loader": Dataset_Load_MNIST
    }
}
ATTACKS = {
    "fgsm": ATTACK_FGSM
}

# Main Functions


# Driver Code