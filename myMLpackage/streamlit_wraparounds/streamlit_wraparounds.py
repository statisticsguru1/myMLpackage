import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from io import StringIO
import pycaret
from pycaret.datasets import get_data
import streamlit as st

from ..general_utility.general_utility import *
from ..Data_preparations.Data_preparations import *
from ..scaling_and_transformation.scaling_and_transformation import *
from ..feature_engineering.feature_engineering import *
from ..feature_selection.feature_selection import *
from ..modelling.modelling import *
from ..prediction.prediction import *

def load_data_stream(stream,file_path, **kwargs):
    """
    Load a data file into a DataFrame.

    Parameters:
    - stream (str): The uploaded streamlit object
    - file_path (str): The path to the data file.
    - **kwargs: Additional keyword arguments to pass to the appropriate read function.

    Returns:
    pandas.DataFrame: The loaded DataFrame.
    """
    try:
        # Determine file extension
        file_ext = file_path.split(".")[-1].lower()

        # Read file based on extension
        if file_ext == "csv":
            df = pd.read_csv(stream, **kwargs)
        elif file_ext == "txt":
            df = pd.read_table(stream, **kwargs)
        elif file_ext in ["xls", "xlsx"]:
            df = pd.read_excel(stream, **kwargs)
        else:
            raise ValueError("Unsupported file format. Please provide a CSV, TXT, XLS, or XLSX file.")
        
        # Strip column names
        df.rename(columns=lambda x: x.strip(), inplace=True)

        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
# A function to gather data information


def dtypess1(data):
    """
    Function to customize variable types detected by the system.

    Parameters:
    - data : pandas DataFrame
        The dataset for which variable types need to be customized.

    Returns:
    - out : dict
        A dictionary containing the customized variable types.
        Keys:
        - 'datatypes': str
            'Yes' if the detected variable types are correct, 'No' otherwise.
        - 'numeric_features': list
            List of numeric features selected by the user.
        - 'categorical_features': list
            List of categorical features selected by the user.
        - 'date_features': list
            List of date features selected by the user.
        - 'create_date_columns': str
            String containing date features to create (separated by space), specified by the user.
        - 'text_features': list
            List of text features selected by the user.
        - 'text_features_method': list
            List of text features methods selected by the user.
        - 'ordinal_features': list
            List of ordinal features selected by the user.
        - 'ignore_features': list
            List of features to be ignored, selected by the user.
        - 'keep_features': list
            List of features to be kept, selected by the user.
    Example:
    >>> import pandas as pd
    >>> data = pd.DataFrame({
    ...     'A': [1, 2, 3],
    ...     'B': ['X', 'Y', 'Z'],
    ...     'C': ['2022-01-01', '2022-02-01', '2022-03-01']
    ... })
    >>> out = dtypess1(data)
    >>> print(out)
    {'datatypes': 'Yes', 'numeric_features': [], 'categorical_features': [], 
    'date_features': [], 'create_date_columns': '', 'text_features': [], 
    'text_features_method': [], 'ordinal_features': [], 'ignore_features': [], 
    'keep_features': []}
    """
    
    datatypes=st.sidebar.radio('Did I detect the variable types correctly?',['Yes','No'])
    if datatypes=='No':
       st.sidebar.write("Change dtypes where needed")
       numeric_features = st.sidebar.multiselect("Numeric Features:", list(data.columns),default=None)
       categorical_features = st.sidebar.multiselect("Categorical Features:",list(data.columns))
       date_features = st.sidebar.multiselect("Date Features:",list(data.columns))
       create_date_columns=st.sidebar.text_input("Date features to create, seperated by space.should be pandas.Series.dt attr:")
       text_features = st.sidebar.multiselect("Text Features:",list(data.columns))
       text_features_method=st.sidebar.multiselect("Text features method:",['bow','tf-idf'])
       ordinal_features=st.sidebar.multiselect("Ordinal Features:",list(data.columns))
       ignore_features = st.sidebar.multiselect("Ignore Features:",list(data.columns))
       keep_features= st.sidebar.multiselect("Keep Features:",list(data.columns))
       out=datypess(datatypes=datatypes,numeric_features=numeric_features,categorical_features=categorical_features,date_features=date_features,text_features=text_features,ordinal_features=ordinal_features,ignore_features=ignore_features,keep_features=keep_features)
    else:
      out=datypess(datatypes=datatypes)
    return out
  
def render_target(data):
    """
    Render a select box for choosing the target column.

    Parameters:
    - data (DataFrame): The DataFrame containing the data.

    Returns:
    - str or None: The selected target column name, or None if no column is selected.

    Example:
    >>> target_column = render_target(data)
    >>> print(target_column)
    'SalePrice'
    """
    target= st.sidebar.selectbox("Target Column Name:", [None] + list(data.columns))
    return target
    
def process_target1(data,target,data_info,sample_data):
    """
    Process the selected target column by handling missing values.Similar to process_target1
    but adds streamlit functionalities like warnings.

    Parameters:
    - data (DataFrame): The DataFrame containing the data.
    - target (str): The name of the selected target column.
    - data_info (st.empty): Streamlit element for displaying data information.
    - sample_data (st.empty): Streamlit element for displaying sample data.

    Returns:
    - list: A list containing the processed DataFrame and the target column name.

    Example:
    >>> processed_data, target_column = process_target1(data, 'SalePrice', data_info, sample_data)
    """
    missing_values_target = data[target].isnull().sum()
    if missing_values_target > 0:
       warning_message =st.sidebar.warning("The selected target column '{}' has {} missing cases.To proceed with this target, the missing cases will be dropped.".format(target, missing_values_target))
       data=process_target(data,target)
       #warning_message.empty()
       data_info.write(calculate_variable_info(data))
       sample_data.write(data.head())
       re=data
    else:
      re=data
    return re
        
def process_missing1(missing):
    """
    Process missing values by allowing the user to configure imputation settings.The function 
    determines whether there is atleast a feature with missing values, if no missing,
    it silently returns the default pycaret imputation settings else it presents users with 
    inputs to configure imputation settings.

    Parameters:
    - missing (pd.Series): A numeric Series indicating the number of missing values in each column of the data.

    Returns:
    - dict: A dictionary containing the configured imputation settings.This can be passed to pycaret setup

    Example:
    >>> imputation_settings = process_missing1(missing_values)
    """
    
    if missing.any() > 0:
       st.sidebar.write("This data has missing values configure imputation settings")
       imputation_type = st.sidebar.selectbox("Imputation type:",['simple','iterative'])
       numeric_imputation= st.sidebar.selectbox("Numeric imputation:",['drop','mean','median','mode','knn','custom'])
       if numeric_imputation=='custom':
          custom_numeric= st.sidebar.number_input("Enter custom value:",min_value=0.0, max_value=None, value=0.0,step=0.01)
       else:
         custom_numeric=0
       categorical_imputation= st.sidebar.selectbox("Categorical imputation:",['drop','mode','custom'])
       if categorical_imputation=='custom':
          custom_categorical=st.sidebar.text_input("Enter custom text:")
       else:
         custom_categorical=""
       iterative_imputation_iters=st.sidebar.number_input("Iterativeimputation iters:",min_value=0, max_value=None, value=5,step=1)
       numeric_iterative_imputer= st.sidebar.selectbox("numeric iterative imputer:",['lightgbm',None])
       categorical_iterative_imputer= st.sidebar.selectbox("categorical iterative imputer:",['lightgbm',None])
       outs=processs_missing(missing,imputation_type=imputation_type,numeric_imputation=numeric_imputation,custom_numeric=custom_numeric,categorical_imputation=categorical_imputation,custom_categorical=custom_categorical,iterative_imputation_iters=iterative_imputation_iters,numeric_iterative_imputer=numeric_iterative_imputer,categorical_iterative_imputer=categorical_iterative_imputer)
    else:
       outs=processs_missing(missing)
    return outs
  
def process_outliers1():
    """
    Process outliers by allowing the user to configure outlier removal settings.

    Returns:
    - dict: A dictionary containing the configured outlier removal settings.

    Example:
    >>> outlier_settings = process_outliers1()
    """
    outliers=st.sidebar.radio('Should outliers be removed?',['Yes','No'],index=1)
    if outliers=='Yes':
       outliers_method= st.sidebar.selectbox("outliers method:",['iforest','ee','lof'])
       thresh=st.sidebar.number_input("outliers threshold:",min_value=0.0, max_value=None, value=0.05,step=0.01)
       out=processs_outliers(remov_outliers=outliers,outliers_method=outliers_method,thresh=thresh)
    else:
       out=processs_outliers(remov_outliers=outliers)
    return out

def fix_imbalances1():
    """
    Function to fix target imbalances in the dataset.

    Returns:
    - out : dict
        A dictionary containing information about fixing target imbalances.
        Keys:
        - 'fix_imbalance': bool
            True if target imbalance is to be fixed, False otherwise.
        - 'fix_imbalance_method': str
            The method chosen to fix the target imbalance (default is 'SMOTE').

    Example:
    >>> out = fix_imbalances1()
    """
    imbalances=st.sidebar.radio('Fix target imbalance?',['Yes','No'],index=1)
    if imbalances=='Yes':
       imbalance_method=st.sidebar.text_input('Feature method:',value='SMOTE')
       out=fix_imbalances(fix_imbalance=True, fix_imbalance_method=imbalance_method)
    else:
      out=fix_imbalances()
    return out

def normalizer1():
    """
    Function to normalize data if chosen by the user.

    Returns:
    - out : dict
        A dictionary containing information about normalization.
        Keys:
        - 'normalize': bool
            True if normalization is chosen, False otherwise.
        - 'normalize_method': str
            The method chosen for normalization (default is 'zscore', other options: 'minmax', 'maxabs', 'robust').

    Example:
    >>> out = normalizer1()
    """
    norm=st.sidebar.radio('Normalize data?',['Yes','No'],index=1)
    if norm=='Yes':
       norm_method=st.sidebar.selectbox("normalize_method:",['zscore','minmax','maxabs','robust'])
       out=normalizer(normalize=True,normalize_method=norm_method)
    else:
       out=normalizer()
    return out
  
def transformations1():
    """
    Function to apply feature transformations if chosen by the user.

    Returns:
    - out : dict
        A dictionary containing information about the feature transformations.
        Keys:
        - 'transformation': bool
            True if feature transformation is chosen, False otherwise.
        - 'transformation_method': str
            The method chosen for feature transformation (default is 'yeo-johnson', other option: 'quantile').

    Example:
    >>> out = transformations1()
    """
    tran=st.sidebar.radio('Transform features?',['Yes','No'],index=1)
    if tran=='Yes':
       tran_method=st.sidebar.selectbox("transformation_method:",['yeo-johnson','quantile'])
       out=transformations(transformation=True,transformation_method=tran_method)
    else:
      out=transformations()
    return out
      
def target_transformation1(data,target,datatypes):
    """
    Function to apply target variable transformation if chosen by the user.

    Args:
    - data : pandas.DataFrame
        The dataset containing the target variable.
    - target : str
        The name of the target variable.
    - datatypes : dict
        A dictionary containing information about the data types of features.

    Returns:
    - out : dict
        A dictionary containing information about the target variable transformation.
        Keys:
        - 'transform_target': bool
            True if target variable transformation is chosen, False otherwise.
        - 'transform_target_method': str
            The method chosen for target variable transformation (default is 'yeo-johnson', other option: 'quantile').

    Example:
    >>> out = target_transformation1(data, 'target_column', {'numeric_features': [], 'categorical_features': []})
    """
    target_tran=st.sidebar.radio('Transform target variable?',['Yes','No'],index=1)
    if target_tran=='Yes':
       tran_method=st.sidebar.selectbox("transformation_method:",['yeo-johnson','quantile'])
       out=target_transformation(data=data, target=target, datatypes=datatypes,transform_target=True,transform_target_method=tran_method)
    else:
      out=target_transformation(data=data, target=target, datatypes=datatypes)
    return out     

def feature_engineering1(data):
    """
    Function to perform feature engineering based on user inputs.

    Args:
    - data : pandas.DataFrame
        The dataset to perform feature engineering on.

    Returns:
    - dict
        A dictionary containing parameters for feature engineering.
        Keys:
        - 'polynomial_features' : bool
            Whether to engineer polynomial features (True if yes, False if no).
        - 'polynomial_degree' : float
            The degree of polynomial features to engineer.
        - 'group_features' : list
            List of features to group.
        - 'drop_groups' : bool
            Whether to drop the groups or not.
        - 'bin_numeric_features' : list
            List of numeric features to bin.
        - 'rare_to_value' : float
            Threshold to consider a value as rare.
        - 'rare_value' : str
            Value to assign for rare values.

    Example:
    >>> feature_params = feature_engineering1(data)
    """
    polyn=st.sidebar.radio('Engineer polynomial features?',['Yes','No'],index=1)
    if polyn=='Yes':
       polynomial_features=True
       polynomial_degree= st.sidebar.number_input("Enter polynomial degree:",min_value=0.0, max_value=None, value=0.0,step=0.01)
    else:
      polynomial_features=False
      polynomial_degree=2
      
    groupfeat=st.sidebar.radio('Extract group features?',['Yes','No'],index=1)
    if groupfeat=='Yes':
       group_names = st.sidebar.text_input("Provide group names separated by comma (,)", value="")
       if group_names!="":
          group_names_list = [name.strip() for name in group_names.split(',')]
          group_features = {}
          for group_name in group_names_list:
              options = st.sidebar.multiselect("Select options for {}:".format(group_name), data.columns)
              group_features[group_name] = options
       else:
          group_features=None
       drop_groups=st.sidebar.radio('drop groups?',[True,False])
    else:
      group_features=None
      drop_groups=False
      
    bin_numeric_features=st.sidebar.multiselect("Bin numeric features:", list(data.columns),default=None)
    rare_to_value=st.sidebar.number_input("rare_to_value:",min_value=0.0, max_value=1.0, value=None,step=0.01)    
    if rare_to_value is not None: 
       rare_value=st.sidebar.text_input("rare value:",value='rare')
    else:
      rare_value='rare'
    return feature_engineering(polynomial_features=polynomial_features, polynomial_degree=polynomial_degree, group_features=group_features, drop_groups=drop_groups, bin_numeric_features=bin_numeric_features, rare_to_value=rare_to_value, rare_value=rare_value)

def feature_select1():
    """
    Function to configure feature selection parameters based on user inputs.

    Returns:
    - dict
        A dictionary containing feature selection parameters.
        Keys:
        - 'feature_selection' : bool
            Whether to perform feature selection (True if yes, False if no).
        - 'feature_selection_method' : str
            The method for feature selection.
        - 'feature_selection_estimator' : str
            The estimator for feature selection.
        - 'n_features_to_select' : int
            The number of features to select.
        - 'remove_multicollinearity' : bool
            Whether to remove multicollinearity (True if yes, False if no).
        - 'multicollinearity_threshold' : float
            The threshold for multicollinearity.
        - 'pca' : bool
            Whether to use PCA dimensionality reduction (True if yes, False if no).
        - 'pca_method' : str
            The method for PCA.
        - 'pca_components' : int
            The number of components for PCA.
        - 'low_variance_threshold' : float
            The threshold for low variance.

    Example:
    >>> feature_params = feature_select1()
    """
    
    featureselect=st.sidebar.radio('Perform feature selection?',['Yes','No'],index=1)
    if featureselect=='Yes':
       feature_selection=True
       feature_selection_method=st.sidebar.selectbox("feature selection method:",['univariate','classic','sequential'])
       feature_selection_estimator=st.sidebar.text_input('feature selection estimator:',value='lightgbm')
       n_features_to_select=st.sidebar.number_input("number of features to select:",min_value=0, max_value=None, value=2,step=1) 
    else:
       feature_selection=False
       feature_selection_method='classic'
       feature_selection_estimator='lightgbm'
       n_features_to_select=2
    multicol=featureselect=st.sidebar.radio('Remove multicollinearity?',['Yes','No'],index=1)
    if multicol=='Yes': 
       remove_multicollinearity=True,
       multicollinearity_threshold=st.sidebar.number_input("multicollinearity threshold:",min_value=0.0, max_value=None, value=0.9,step=0.01) 
    else:
       remove_multicollinearity=False,
       multicollinearity_threshold=0.9
    pcadim=st.sidebar.radio('Use pca dimensionality reduction?',['Yes','No'],index=1)
    if pcadim=='Yes': 
       pca=True
       pca_method=feature_selection_method=st.sidebar.selectbox("pca method:",['linear','kernel','incremental'])
       pca_components=st.sidebar.number_input("pca components:",min_value=0, max_value=None, value=4,step=1) 
    else:
       pca=False
       pca_method='linear'
       pca_components=4
    low_variance_threshold=st.sidebar.number_input("low variance threshold:",min_value=0.0, max_value=None, value=None,step=0.01)
    return feature_select(feature_selection=feature_selection, feature_selection_method=feature_selection_method, feature_selection_estimator=feature_selection_estimator, n_features_to_select=n_features_to_select, remove_multicollinearity=remove_multicollinearity, multicollinearity_threshold=multicollinearity_threshold, pca=pca, pca_method=pca_method, pca_components=pca_components,low_variance_threshold=low_variance_threshold)

def prediction1():
    st.header("Predictions")
    st.write("Upload Prediction Dataset")
    newfile_file = st.file_uploader("Browse data")
    if newfile_file is not None:
       pred=prediction(cat_tune,target,newfile_file)
       st.write("predictions:",settingsinfo)
       
