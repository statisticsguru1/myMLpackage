import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from io import StringIO
import pycaret
from pycaret.datasets import get_data
import streamlit as st
def datypess(datatypes='No',numeric_features=[],categorical_features=[],date_features=[],text_features=[],ordinal_features=[],ignore_features=[],keep_features=[]):
    """
    Constructs a data dictionary with the correct data types provided by user. The dict can be used to overwrite
    wrongly inffered dtypes during pycaret setup

    Parameters:
    - data (DataFrame): The input data containing the variables for which data types need to be fixed.
    - datatypes (str): Indicates whether all data columns are correctly inffered it takes:
      'Yes' : to mean all dtypes are correct and no further correction is needed
      'No':Atleast one data type is wrongly inffered
    - numeric_features (list): A list of numeric feature names if specified by the user, otherwise an empty list.
    - categorical_features (list): A list of categorical feature names if specified by the user, otherwise an empty list.
    - date_features (list): A list of date feature names if specified by the user, otherwise an empty list.
    - text_features (list): A list of text feature names if specified by the user, otherwise an empty list.
    - ordinal_features (list): A list of ordinal feature names if specified by the user, otherwise an empty list.
    - ignore_features (list): A list of features to be ignored if specified by the user, otherwise an empty list.
    - keep_features (list): A list of features to be kept if specified by the user, otherwise an empty list.

    Returns:
    dict: A dictionary containing information about the determined data types of variables.
    """
    if datatypes=='No':
       dtypz={
         'numeric_features':numeric_features,
         'categorical_features':categorical_features,
         'date_features' :date_features,
         'text_features':text_features,
         'ordinal_features':ordinal_features,
         'ignore_features':ignore_features,
         'keep_features':keep_features
       }
    else:
       dtypz={
         'numeric_features':None,
         'categorical_features':None,
         'date_features':None,
         'text_features':None,
         'ordinal_features':None,
         'ignore_features':None,
         'keep_features':None
       }
    return dtypz
def determine_model_type(data,target,dtyps):
    """
    Determine the type of model based on the target variable and data types.

    Parameters:
    - data (DataFrame): The input dataset.
    - target (str): The name of the target variable.
    - dtyps (dict): A dictionary containing information about the data types of features.

    Returns:
    str: The type of model, either 'classification' or 'regression', based on the target variable and data types.
    """
    if target in  (dtyps['categorical_features'] or []) or data[target].dtype == 'object':
       mod='classification'
    else:
      mod='regression'
    return mod
def process_target(data,target):
    """
    Process the target variable by handling missing values.

    Parameters:
    - data (DataFrame): The input data containing the target variable.
    - target (str): The name of the target column.

    Returns:
    list: A list containing the processed data and the target variable.
    """
    missing_values_target = data[target].isnull().sum()
    if missing_values_target > 0:
       data.dropna(subset=[target], inplace=True)
       re=data
    else:
      re=data
    return re
def processs_missing(missing,imputation_type=None,numeric_imputation='drop',custom_numeric=5,categorical_imputation='drop',custom_categorical="",iterative_imputation_iters=5,numeric_iterative_imputer=None,categorical_iterative_imputer=None):
    """
    Process missing values by allowing the user to configure imputation settings.The function 
    determines whether there is atleast a feature with missing values, if no missing,
    it silently returns the default pycaret imputation settings else it presents users with 
    inputs to configure imputation settings.

    Parameters:
    - missing (DataFrame): A numeric Series indicating the number of missing values in each column of the data
    - imputation_type (str): The type of imputation to use ('simple' or 'iterative'). Default is None.
    - numeric_imputation (str): The method for numeric imputation.One of {'drop','mean','median','mode','knn','custom'}. Default is 'drop'.
    - custom_numeric (float): The custom value to use for numeric imputation when 'custom' method is selected.
    - categorical_imputation (str): The method for categorical imputation.One of {'drop','mode','custom'}. Default is 'drop'.
    - custom_categorical (str): The custom value to use for categorical imputation when 'custom' method is selected..
    - iterative_imputation_iters (int): The number of iterations for iterative imputation. Default is 5.
    - numeric_iterative_imputer (str): The method for numeric iterative imputation. Default is None.
    - categorical_iterative_imputer (str): The method for categorical iterative imputation. Default is None.

    Returns:
    dict: A dictionary containing the configured imputation settings.
    """

    if missing.any() > 0:
       imputes={
         'imputation_type':imputation_type,
         'numeric_imputation':numeric_imputation,
         'categorical_imputation':categorical_imputation,
         'iterative_imputation_iters':iterative_imputation_iters,
         'numeric_iterative_imputer':numeric_iterative_imputer,
         'categorical_iterative_imputer':categorical_iterative_imputer
       }
       
       if numeric_imputation == 'custom':
          imputes['numeric_imputation'] = custom_numeric

       # Conditionally update the dictionary if categorical_imputation is 'custom'
       if categorical_imputation == 'custom':
          imputes['categorical_imputation'] = custom_categorical
    else:
      imputes={
        'imputation_type':None,
         'numeric_imputation':'drop',
         'categorical_imputation':'drop',
         'iterative_imputation_iters':5,
         'numeric_iterative_imputer':None,
         'categorical_iterative_imputer':None
       }
    return imputes
def processs_outliers(remov_outliers='Yes',outliers_method="iforest",thresh=0.05):
    """
    Process outliers in the dataset and configure outlier removal settings.
    Parameters:
    - remov_outliers (str): Indicates whether to remove outliers or not. Options are 'Yes' or 'No'. Default is 'False'.
    - outliers_method (str): The method for outlier detection. Default is "iforest".
    - thresh (float): The threshold for outlier detection. Default is 0.05.

    Returns:
    dict: A dictionary containing the configured outlier removal settings.
    """
    
    if remov_outliers=='Yes':
       outlier={
         'remove_outliers':True,
         'outliers_method':outliers_method,
         'outliers_threshold':thresh
         }
    else:
      outlier={
        'remove_outliers':False,
        'outliers_method':"iforest",
        'outliers_threshold':0.05
        }
    return outlier
def fix_imbalances(fix_imbalance=False, fix_imbalance_method=None):
    """
    Configure settings to fix class imbalance in the dataset.

    Parameters:
    - fix_imbalance (bool): Indicates whether to fix class imbalance or not. Default is False.
    - fix_imbalance_method (str): The method to fix class imbalance. Default is None.

    Returns:
    dict: A dictionary containing the configured settings to fix class imbalance.
    """
    out = {
        'fix_imbalance': fix_imbalance,
        'fix_imbalance_method': fix_imbalance_method
    }
    return out
