import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from io import StringIO
import pycaret
from pycaret.datasets import get_data
import streamlit as st
def feature_engineering(polynomial_features=False, polynomial_degree=2, group_features=None, drop_groups=False, bin_numeric_features=None, rare_to_value=None, rare_value='rare'):
    """
    Configure feature engineering settings.

    Parameters:
    - polynomial_features (bool): Whether to create polynomial features based on numeric features. Default is False.
    - polynomial_degree (int): Degree of polynomial features to be created. Default is 2.
    - group_features (list or list of list): List of features to be grouped for statistical feature extraction. Default is None.
    - drop_groups (bool): Whether to drop the original features in the group. Ignored when ``group_features`` is None.
    - bin_numeric_features (list): List of numeric features to be binned into intervals. Default is None.
    - rare_to_value (float or None): Minimum fraction of category occurrences in a categorical column to be considered rare. Default is None.
    - rare_value (str): Value to replace rare categories with. Default is 'rare'.

    Returns:
    dict: A dictionary containing the configured feature engineering settings.
    """
    if group_features is None:
        group_names = None
        
    out = {
        'polynomial_features': polynomial_features,
        'polynomial_degree': polynomial_degree,
        'group_features': group_features,
        'drop_groups': drop_groups,
        'bin_numeric_features': bin_numeric_features,
        'rare_to_value': rare_to_value,
        'rare_value': rare_value
    }
    return out
