"""
Module: streamlit_wraparounds

Description:
This module provides wraparounds of various functions to add Streamlit functionalities 
such as inputs (text, numeric, radio buttons, etc.) to streamline the machine learning 
pipeline setup process. It includes functions for loading data, customizing variable types, 
rendering target columns, processing target variables, handling missing values, processing 
outliers, fixing target imbalances, normalizing data, applying feature transformations, 
performing feature engineering, configuring feature selection, and making predictions.

Functions:
- load_data_stream: Load a data file into a DataFrame.
- dtypess1: Customize variable types detected by the system.
- render_target: Render a select box for choosing the target column.
- process_target1: Process the selected target column by handling missing values.
- process_missing1: Process missing values by allowing the user to configure imputation settings.
- process_outliers1: Process outliers by allowing the user to configure outlier removal settings.
- fix_imbalances1: Fix target imbalances in the dataset.
- normalizer1: Normalize data if chosen by the user.
- transformations1: Apply feature transformations if chosen by the user.
- target_transformation1: Apply target variable transformation if chosen by the user.
- feature_engineering1: Perform feature engineering based on user inputs.
- feature_select1: Configure feature selection parameters based on user inputs.
- prediction1: Upload a prediction dataset and make predictions.

Usage:
Import the module and use the provided functions to interactively configure machine 
learning pipeline settings in a Streamlit app.

Example:
```python
import streamlit_wraparounds

# Load data into a DataFrame
data = streamlit_wraparounds.load_data_stream(stream, file_path)

# Customize variable types detected by the system
variable_types = streamlit_wraparounds.dtypess1(data)

# Render a select box for choosing the target column
target_column = streamlit_wraparounds.render_target(data)

# Process the selected target column by handling missing values
processed_data = streamlit_wraparounds.process_target1(data, target_column)
"""
from .streamlit_wraparounds import *
