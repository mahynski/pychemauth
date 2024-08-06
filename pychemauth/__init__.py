"""
Load all modules.

author: nam
"""
__version__ = "0.0.0-beta4"

__all__ = [
    "analysis",
    "classifier",
    "eda",
    "manifold",
    "preprocessing",
    "regressor",
]

# For Keras 3 - PyChemAuth uses a tensorflow backend
# https://keras.io/getting_started/#installing-keras-3
import os

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["CUDA_VISIBLE_DEVICES"] = "" # Manually disable GPU
