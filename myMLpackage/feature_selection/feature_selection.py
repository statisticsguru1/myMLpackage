import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from io import StringIO
import pycaret
from pycaret.datasets import get_data
import streamlit as st
# feature selection   
def feature_select(feature_selection=False, feature_selection_method='classic', feature_selection_estimator='lightgbm', n_features_to_select=0.2, remove_multicollinearity=False, multicollinearity_threshold=0.9, pca=False, pca_method='linear', pca_components=None, low_variance_threshold=None):
    """
    Configure feature selection settings.

    Parameters:
    - feature_selection (bool): Whether to perform feature selection. Default is False.
    - feature_selection_method (str): Method for feature selection. Choose from 'univariate', 'classic', or 'sequential'. Default is 'classic'.
    - feature_selection_estimator (str or sklearn estimator): Estimator used for determining feature importance. Default is 'lightgbm'.
    - n_features_to_select (int or float): Maximum number of features to select. Default is 0.2.
    - remove_multicollinearity (bool): Whether to remove multicollinear features. Default is False.
    - multicollinearity_threshold (float): Threshold for identifying multicollinear features. Default is 0.9.
    - pca (bool): Whether to apply PCA for dimensionality reduction. Default is False.
    - pca_method (str): Method for PCA. Choose from 'linear', 'kernel', or 'incremental'. Default is 'linear'.
    - pca_components (int, float, str, or None): Number of components to keep for PCA. Default is None.
    - low_variance_threshold (float or None): Threshold for removing low variance features. Default is None.

    Returns:
    dict: A dictionary containing the configured feature selection settings.
    """
    out = {
        'feature_selection': feature_selection,
        'feature_selection_method': feature_selection_method,
        'feature_selection_estimator': feature_selection_estimator,
        'n_features_to_select': n_features_to_select,
        'remove_multicollinearity': remove_multicollinearity,
        'multicollinearity_threshold': multicollinearity_threshold,
        'pca': pca,
        'pca_method': pca_method,
        'pca_components': pca_components,
        'low_variance_threshold': low_variance_threshold
    }
    return out
