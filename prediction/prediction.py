import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from io import StringIO
import pycaret
from pycaret.datasets import get_data
import streamlit as st
# prediction
def prediction(model,target,new_filepath):
    """
    Generate predictions using the specified model and dataset.

    Parameters:
    - model (object): The trained model used for making predictions.
    - target (str): The name of the target variable.
    - new_filepath (str): The file path to the dataset for which predictions are to be made.
    - model_type (str): The type of model, either 'classification' or 'regression'.

    Returns:
    DataFrame or None: A DataFrame containing the predictions if successful, otherwise None.
    """    
    try:
      # Determine file extension
      file_ext = new_filepath.split(".")[-1].lower()
      # Read file based on extension
      if file_ext == "csv":
         df = pd.read_csv(new_filepath)
      elif file_ext == "txt":
         df = pd.read_table(new_filepath)
      elif file_ext in ["xls", "xlsx"]:
         df = pd.read_excel(new_filepath)
      else:
        raise ValueError("Unsupported file format. Please provide a CSV, TXT, XLS, or XLSX file.")
      # Strip column names
      df.rename(columns=lambda x: x.strip(), inplace=True)
      if target in df.columns:
         df.rename(columns={target:target+'_zipper'}, inplace=True)
         if model_type=='classification':
            pred=classification.predict_model(model,data=df)
            pred.rename(columns={target+'_zipper':target}, inplace=True)
         else:
           pred=regression.predict_model(model,data=df)
           pred.rename(columns={target+'_zipper':target}, inplace=True)
         return pred
    except Exception as e:
           print(f"Error loading data: {e}")
           return None
