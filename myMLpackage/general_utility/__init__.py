"""
General Utility Module
======================

This module provides general utility functions for data processing, visualization, and analysis.

Functions:
    load_data(file_path, **kwargs): Load a data file into a DataFrame.
    calculate_variable_info(data): Calculate information about variables in the DataFrame.
    plot_histogram(data, column, bins=50): Plot a histogram for a specified column in the dataset.
    plot_countplot(data, x, hue=None, xlabel=None, ylabel=None, title=None): Plot a countplot for the specified column in the dataset.
    plot_correlation_matrix(data, dtyps): Plot a correlation matrix heatmap for numeric columns in the dataset.

Usage Example:
--------------
Import the module:

    import myMLpackage.general_utility as gu

Load a dataset:

    file_path = "data.csv"
    data = gu.load_data(file_path)

Calculate variable information:

    variable_info = gu.calculate_variable_info(data)
    print(variable_info)

Plot a histogram:

    gu.plot_histogram(data, 'Age')

Plot a countplot:

    gu.plot_countplot(data, x='Sex', hue='Survived', xlabel='Gender', ylabel='Count', title='Survival by Gender')

Plot a correlation matrix heatmap:

    gu.plot_correlation_matrix(data, dtyps={'numeric_features': ['Age', 'Fare'], 'categorical_features': ['Sex']})

"""
from .general_utility import *