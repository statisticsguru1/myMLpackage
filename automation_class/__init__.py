"""
Module: automation_class

Description:
This module contains a class called Modelling that provides functionality for automating the machine learning modeling process. 
The Modelling class encapsulates methods for loading data, calculating variable information, determining data types, 
handling missing values, processing outliers, fixing class imbalances, normalization, transformations, target transformation, 
feature engineering, feature selection, setting up PyCaret, comparing models, tuning the best model, generating predictions, 
and visualizing results.

Classes:
- Modelling: A class that automates the machine learning modeling process using PyCaret.

Methods:
- load_data: Load a data file into a DataFrame.
- calculate_variable_info: Calculate information about variables in the given DataFrame.
- datypess: Constructs a data dictionary with the correct data types provided by the user.
- determine_model_type: Determine the type of model based on the target variable and data.
- processs_missing: Process missing values in the dataset.
- processs_outliers: Process outliers in the dataset and configure outlier removal settings.
- fix_imbalances: Configure settings to fix class imbalance in the dataset.
- normalizer: Configure normalization settings for the feature space transformation.
- transformations: Configure transformation settings for the data to make it more normal/Gaussian-like.
- target_transformation: Configure target transformation settings.
- feature_engineering: Configure feature engineering settings.
- feature_select: Configure feature selection settings.
- setup_pycaret: Setup PyCaret for classification or regression based on the provided data and settings.
- model_comparisons: Compare different models and select the best one based on the dataset.
- tuned_model_results: Tune the selected model and display the results.
- prediction: Generate predictions for new data using the tuned model.
- graph_results: Visualize the results of the tuned model.
- plot_histogram: Plot a histogram for a specified column in the dataset.
- plot_countplot: Plot a countplot for the specified column in the dataset.
- plot_correlation_matrix: Plot a correlation matrix heatmap for numeric columns in the dataset.
- data_head: Display the first n rows of the dataset.

Example:
    # Import the module
    import automation_class
    
    # Initialize the Modelling object
    model = automation_class.Modelling(file_path='data.csv', target='target_column')
    
    # Load data
    data = model.load_data()
    
    # Calculate variable information
    variable_info = model.calculate_variable_info()
    
    # Determine data types
    datatypes = model.datypess()
    
    # Determine model type
    model_type = model.determine_model_type()
    
    # Process missing values
    missing_values_info = model.processs_missing()
    
    # Process outliers
    outliers_info = model.processs_outliers()
    
    # Fix class imbalances
    imbalance_info = model.fix_imbalances()
    
    # Configure normalization
    normalization_info = model.normalizer()
    
    # Configure transformations
    transformation_info = model.transformations()
    
    # Configure target transformation
    target_transformation_info = model.target_transformation()
    
    # Configure feature engineering
    feature_engineering_info = model.feature_engineering()
    
    # Configure feature selection
    feature_selection_info = model.feature_select()
    
    # Setup PyCaret
    setup_info = model.setup_pycaret()
    
    # Compare models
    comparison_info = model.model_comparisons()
    
    # Tune the best model
    tuned_model_info = model.tuned_model_results()
    
    # Generate predictions
    predictions = model.prediction(new_filepath='new_data.csv')
    
    # Visualize results
    visualization = model.graph_results()
    
    # Plot histogram
    histogram = model.plot_histogram()
    
    # Plot countplot
    countplot = model.plot_countplot()
    
    # Plot correlation matrix
    correlation_matrix = model.plot_correlation_matrix()
    
    # Display data head
    data_head = model.data_head()
"""

from .automation_class import *