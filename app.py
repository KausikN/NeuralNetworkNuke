"""
Stream lit GUI for hosting {FEATUREDATA_PROJECT_NAME}
"""

# Imports
import os
import streamlit as st
import json
import numpy as np

import NeuralNetworkNuke

# Main Vars
config = json.load(open("./StreamLitGUI/UIConfig.json", "r"))

# Main Functions
def main():
    # Create Sidebar
    selected_box = st.sidebar.selectbox(
    "Choose one of the following",
        tuple(
            [config["PROJECT_NAME"]] + 
            config["PROJECT_MODES"]
        )
    )
    
    if selected_box == config["PROJECT_NAME"]:
        HomePage()
    else:
        correspondingFuncName = selected_box.replace(" ", "_").lower()
        if correspondingFuncName in globals().keys():
            globals()[correspondingFuncName]()
 

def HomePage():
    st.title(config["PROJECT_NAME"])
    st.markdown("Github Repo: " + "[" + config["PROJECT_LINK"] + "](" + config["PROJECT_LINK"] + ")")
    st.markdown(config["PROJECT_DESC"])

    # st.write(open(config["PROJECT_README"], "r").read())

#############################################################################################################################
# Repo Based Vars
CACHE_PATH = "StreamLitGUI/CacheData/Cache.json"

# Util Vars
CACHE = {}

# Util Functions
def LoadCache():
    global CACHE
    CACHE = json.load(open(CACHE_PATH, "r"))

def SaveCache():
    global CACHE
    json.dump(CACHE, open(CACHE_PATH, "w"), indent=4)

# Main Functions


# UI Functions
def UI_LoadDatasetSubset_MNIST(DATASET):
    random_index = np.random.randint(0, DATASET["X_test"].shape[0])
    RANDOM_IMAGE = {
        "X": DATASET["X_test"][random_index],
        "Y": DATASET["Y_test"][random_index]
    }
    return RANDOM_IMAGE

# Repo Based Functions
def fgsm_attack():
    # Title
    st.header("FGSM Attack")

    # Prereq Loaders

    # Load Inputs
    USERINPUT_Model = st.selectbox("Choose a model", list(NeuralNetworkNuke.MODELS.keys()))

    # Process Inputs
    if st.button("Attack"):
        # Load Model
        MODEL_DATA = NeuralNetworkNuke.MODELS[USERINPUT_Model]
        MODEL = NeuralNetworkNuke.Model_LoadPickle(MODEL_DATA["model_path"])
        # Load Dataset
        DATASET = MODEL_DATA["dataset_loader"]()
        RANDOM_IMAGE = UI_LoadDatasetSubset_MNIST(DATASET)
        # Run Attack
        ATTACK_DATA = NeuralNetworkNuke.ATTACKS["fgsm"][USERINPUT_Model]["display"](
            MODEL, RANDOM_IMAGE["X"], RANDOM_IMAGE["Y"],
            y_target=None, eps=0.25, cmap="Greys"
        )
        # Display Outputs
        st.plotly_chart(ATTACK_DATA["plot"])
    
#############################################################################################################################
# Driver Code
if __name__ == "__main__":
    main()