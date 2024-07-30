import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from io import StringIO
import pycaret
from pycaret.datasets import get_data
from pycaret import classification,regression

import streamlit as st

# get model type
def model_typ(data,target,dtyps):
    if target in  (dtyps['categorical_features'] or []) or data[target].dtype == 'object':
       out='classification'
    else:
      out='regression'
    return out
  
       
# Model set up and config 
def setup_pycaret(data, target,use_gpu=False, *args):
    """
    Setup PyCaret for classification or regression based on the provided data and settings.

    Parameters:
    - data (DataFrame): The input dataset.
    - target (str): The target variable.
    - *args (dict): Additional dictionaries containing settings for PyCaret setup.

    Returns:
    object: PyCaret setup object for classification or regression.
    """
    # Combine data and target into a dictionary
    targetanddatadict = {'data': data, 'target': target,'use_gpu':use_gpu}
    
    # Combine all dictionaries into a single master dictionary
    master_dict = targetanddatadict.copy()
    for d in args:
        master_dict.update(d)
    
    # Setup PyCaret based on the provided dictionaries
    if target in  (master_dict['categorical_features'] or []) or data[target].dtype == 'object':
       del master_dict['transform_target']
       del master_dict['transform_target_method']
       setting = classification.setup(**master_dict)
    else:
      del master_dict['fix_imbalance']
      del master_dict['fix_imbalance_method']
      setting = regression.setup(**master_dict)
    return setting

# best model
def model_comparisons(model_type):
    """
    Compare different models and select the best one based on the specified model type.

    Parameters:
    - model_type (str): The type of model, either 'classification' or 'regression'.

    Returns:
    object: The best model selected based on the comparisons.
    """
    if model_type=='classification':
       best_model=classification.compare_models()
    else:
      best_model=regression.compare_models()
    return best_model

# tuned model
def tuned_model(best_model,model_type):
    """
    Tune the selected model based on the specified model type.

    Parameters:
    - best_model (object): The best model selected based on comparisons.
    - model_type (str): The type of model, either 'classification' or 'regression'.

    Returns:
    object: The tuned model with improved performance.
    """
    if model_type=='classification':
       cat = classification.create_model(best_model)
       cat_tune = classification.tune_model(cat)
    else:
      cat = regression.create_model(best_model)
      cat_tune = regression.tune_model(cat)
    return cat_tune

# Graph  results
def graph_results(tuned_model,model_type):
    """
    Plot visualizations of the tuned model's performance.

    Returns:
    matplotlib.pyplot.figure: The matplotlib figure object containing the visualizations.
    """
    if self.model_type=='classification':
       return classification.plot_model(tuned_model)
    else:
      return regression.plot_model(tuned_model)
