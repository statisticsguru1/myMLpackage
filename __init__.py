"""
myMLpackage
===========

A Python package for machine learning tasks, including data preparations, modeling, general utility functions, feature engineering, feature selection, and streamlit wraparounds.

Modules:
--------
- Data_preparations: Module for data preparation tasks such as handling data types, processing the target variable, and dealing with missing values and outliers.
- modeling: Module for setting up PyCaret for classification or regression based on the provided data and settings.
- general_utility: Module providing general utility functions for data processing, visualization, and analysis.
- feature_engineering: Module for configuring feature engineering settings, which can be used in PyCaret set up.
- feature_selection: Module for configuring feature selection settings for PyCaret classification or regression setup.
- streamlit_wraparounds: Module providing wraparounds of various functions to add Streamlit functionalities to streamline the machine learning pipeline setup process.

Usage:
------
Import the package and its modules to utilize the provided functionalities.

Example:
--------
```python
import myMLpackage

# Load data
data = myMLpackage.Data_preparations.load_data('data.csv')

# Setup PyCaret
setup = myMLpackage.modeling.setup_pycaret(data)

# Perform feature engineering
features = myMLpackage.feature_engineering.feature_engineering(polynomial_features=True)

# Make predictions using Streamlit wraparounds
predictions = myMLpackage.streamlit_wraparounds.prediction1(model, 'target_column', 'new_dataset.csv')
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from io import StringIO
import pycaret
from pycaret.datasets import get_data
import streamlit as st
from pycaret import classification,regression
