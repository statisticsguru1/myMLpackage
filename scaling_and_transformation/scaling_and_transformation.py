import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from io import StringIO
import pycaret
from pycaret.datasets import get_data
import streamlit as st
def normalizer(normalize=False,normalize_method='zscore'):
    """
    Configures normalization settings for the feature space transformation.

    Parameters:
    - normalize (bool): Whether to normalize the feature space or not. Default is False.
    - normalized_method (str): The method used for normalization. Default is 'zscore'.
        Available options:
            - 'zscore': Standard z-score normalization.
            - 'minmax': Min-Max normalization, scales features to a range of 0 to 1.
            - 'maxabs': MaxAbs normalization, scales features to the range of -1 to 1 without shifting the mean.
            - 'robust': Robust normalization, scales features according to the Interquartile range.

    Returns:
    dict: A dictionary containing the configured normalization settings.
    """
    out={
      'normalize':normalize,
      'normalize_method':normalize_method
      }
    return out
def transformations(transformation=False, transformation_method='yeo-johnson'):
    """
    Configures transformation settings for the data to make it more normal/Gaussian-like.

    Parameters:
    - transformation (bool): Whether to apply a power transformer to the data or not. Default is False.
    - transformation_method (str): The method used for transformation. Default is 'yeo-johnson'.
        Available options:
            - 'yeo-johnson': Yeo-Johnson transformation, estimates the optimal parameter for stabilizing variance
              and minimizing skewness through maximum likelihood.
            - 'quantile': Quantile transformation, transforms the feature set to follow a Gaussian-like or normal
              distribution. It is non-linear and may distort linear correlations between variables measured at the
              same scale.

    Returns:
    dict: A dictionary containing the configured transformation settings.
    """
    out = {
        'transformation': transformation,
        'transformation_method': transformation_method
    }
    return out
def target_transformation(data, target, datatypes, transform_target=False, transform_target_method='yeo-johnson'):
    """
    Configure target transformation settings.

    Parameters:
    - data (DataFrame): The input dataset.
    - target (str): The target variable.
    - datatypes (dict): A dictionary containing information about the data types of features.
    - transform_target (bool): Whether to apply transformation to the target variable or not. Default is False.
    - transform_target_method (str): The method used for transforming the target variable. Default is 'yeo-johnson'.

    Returns:
    dict: A dictionary containing the configured target transformation settings.
    """
    if target in (datatypes['categorical_features'] or []) or data[target].dtype == 'object':
        transform_target = False
    
    out = {
        'transform_target': transform_target,
        'transform_target_method': transform_target_method
    }
    return out
