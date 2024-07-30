import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from io import StringIO
import pycaret
from pycaret.datasets import get_data
import streamlit as st
# General utility function

# function to load data 
def load_data(file_path, **kwargs):
    """
    Load a data file into a DataFrame.

    Parameters:
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
            df = pd.read_csv(file_path, **kwargs)
        elif file_ext == "txt":
            df = pd.read_table(file_path, **kwargs)
        elif file_ext in ["xls", "xlsx"]:
            df = pd.read_excel(file_path, **kwargs)
        else:
            raise ValueError("Unsupported file format. Please provide a CSV, TXT, XLS, or XLSX file.")
        
        # Strip column names
        df.rename(columns=lambda x: x.strip(), inplace=True)

        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
# A function to gather data information

def calculate_variable_info(data):
    """
    Calculate information about variables in the given DataFrame.

    Parameters:
    - data (DataFrame): The DataFrame containing the variables.

    Returns:
    - DataFrame: A DataFrame containing information about each variable.
                 The DataFrame has the following columns:
                 - Data Type: The data type of each variable.
                 - Unique Values Count: The count of unique values for each variable.
                 - Missing Values Count: The count of missing values for each variable.
                 - Missing Values Percentage: The percentage of missing values for each variable.
                 - Additional columns for numerical variables:
                   - min: The minimum value.
                   - max: The maximum value.
                   - mean: The mean value.
                   - std: The standard deviation.
                 - Additional columns for categorical variables:
                   - Unique Values: The unique values.

    Example:
    >>> data_info = calculate_variable_info(data)
    >>> print(data_info)
               Data Type  Unique Values Count  Missing Values Count  Missing Values Percentage    min  max  mean  std Unique Values
    variable1     int64                   10                      0                        0.0   1.00   10   5.5  3.0           NaN
    variable2   float64                    5                      2                       20.0   2.50    7   4.5  1.5           NaN
    variable3    object                    2                      0                        0.0    NaN  NaN   NaN  NaN    [A, B, C]
    """
    variable_info = pd.DataFrame(index=data.columns)
    variable_info['Data Type'] = data.dtypes
    variable_info['Unique Values Count'] = data.nunique()
    variable_info['Missing Values Count'] = data.isnull().sum()
    variable_info['Missing Values Percentage'] = (variable_info['Missing Values Count'] / len(data)) * 100
    variable_info['Missing Values Percentage'] = variable_info['Missing Values Percentage'].round(2)

    # Handling numerical variables
    numeric_cols = data.select_dtypes(include=np.number).columns
    if not numeric_cols.empty:
        numeric_info = data[numeric_cols].describe().T[['min', 'max', 'mean', 'std']]
        variable_info = pd.concat([variable_info, numeric_info], axis=1, sort=False)

    # Handling categorical variables
    categorical_cols = data.select_dtypes(include='object').columns
    if not categorical_cols.empty:
        for col in categorical_cols:
            variable_info.at[col, 'Unique Values'] = data[col].unique()

    return variable_info
  
# this fun graphs hist for target

def plot_histogram(data, column, bins=50):
    """
    Plot a histogram for a specified column in the dataset.

    Parameters:
    - data (DataFrame): The input dataset.
    - column (str): The column to plot the histogram for.
    - bins (int, optional): The number of bins for the histogram. Defaults to 50.

    Returns:
    matplotlib.pyplot.figure: The matplotlib figure object containing the histogram plot.
    """
    plt.figure(figsize=(15, 10))
    sns.histplot(data=data, x=column, bins=bins, cbar=True)
    plt.title(f"Histogram of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    return plt

# this fun graphs countplot for target    
def plot_countplot(data, x, hue=None, xlabel=None, ylabel=None, title=None):
    """
    Plot a countplot for the specified column in the dataset.

    Parameters:
    - data (DataFrame): The input dataset.
    - x (str): The column to plot on the x-axis.
    - hue (str, optional): The column to differentiate by color.
    - xlabel (str, optional): The label for the x-axis.
    - ylabel (str, optional): The label for the y-axis.
    - title (str, optional): The title of the plot.

    Returns:
    matplotlib.pyplot.figure: The matplotlib figure object containing the countplot.
    """
    plt.figure(figsize=(10, 6))
    sns.countplot(data=data, x=x, hue=hue)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    return plt
  
def plot_correlation_matrix(data, dtyps):
    """
    Plot a correlation matrix heatmap for numeric columns in the dataset.

    Parameters:
    - data (DataFrame): The input dataset.
    - dtyps (dict): A dictionary containing information about data types.

    Returns:
    matplotlib.pyplot.figure: The matplotlib figure object containing the correlation matrix heatmap.
    """
    numeric_cols = [col for col in data.columns if col not in (dtyps['categorical_features'] or []) and data[col].dtype != 'object']
    non_numeric_cols = [col for col in (dtyps['numeric_features'] or []) if col in data.columns and data[col].dtype == 'object']
    selected_cols = numeric_cols + non_numeric_cols
    filtered_data = data[selected_cols]
    for col in non_numeric_cols:
        filtered_data[col] = pd.to_numeric(filtered_data[col], errors='coerce')
    correlation_matrix = filtered_data.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    return plt

def load_experimental_dataset(dataset_name):
    """
    Load a dataset from the 'data' directory within the package.

    Args:
        dataset_name (str): The name of the dataset to load (excluding the file extension).

    Returns:
        pandas.DataFrame: The loaded dataset as a pandas DataFrame.

    Raises:
        FileNotFoundError: If the specified dataset file does not exist.
    """
    # Construct the path to the dataset file
    data_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'data')
    dataset_file = f"{dataset_name}.csv"  # Assuming datasets are CSV files
    dataset_path = os.path.join(data_dir, dataset_file)
    # Check if the dataset file exists
    if os.path.exists(dataset_path):
        # Read the dataset into a pandas DataFrame
        df=pd.read_csv(dataset_path)
        df.rename(columns=lambda x: x.strip(), inplace=True)
        return df
    else:
        # Raise an error if the dataset file does not exist
        raise FileNotFoundError(f"Dataset '{dataset_name}' not found.")
