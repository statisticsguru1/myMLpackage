import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from io import StringIO
import pycaret
from pycaret.datasets import get_data
import streamlit as st
from pycaret import regression,classification
pd.set_option('display.max_columns', None)

class Modelling:
    """
    A class for performing data preprocessing, model training, and evaluation using PyCaret for classification or regression tasks.

    Attributes:
    - data (pd.Dataframe): The data to be used.
    - target (str): The name of the target variable.
    - datatypes (str): Indicates whether all data columns are correctly inferred.
    - numeric_features (list): A list of numeric feature names.
    - categorical_features (list): A list of categorical feature names.
    - date_features (list): A list of date feature names.
    - text_features (list): A list of text feature names.
    - ordinal_features (list): A list of ordinal feature names.
    - ignore_features (list): A list of features to be ignored.
    - keep_features (list): A list of features to be kept.
    - imputation_type (str): The type of imputation to use.
    - numeric_imputation (str): The method for numeric imputation.
    - custom_numeric (float): The custom value to use for numeric imputation.
    - categorical_imputation (str): The method for categorical imputation.
    - custom_categorical (str): The custom value to use for categorical imputation.
    - iterative_imputation_iters (int): The number of iterations for iterative imputation.
    - numeric_iterative_imputer (str): The method for numeric iterative imputation.
    - categorical_iterative_imputer (str): The method for categorical iterative imputation.
    - remov_outliers (str): Indicates whether to remove outliers.
    - outliers_method (str): The method for outlier detection.
    - thresh (float): The threshold for outlier detection.
    - fix_imbalance (bool): Indicates whether to fix class imbalance.
    - fix_imbalance_method (str): The method to fix class imbalance.
    - normalize (bool): Whether to normalize the feature space.
    - normalize_method (str): The method used for normalization.
    - transformation (bool): Whether to apply a power transformer to the data.
    - transformation_method (str): The method used for transformation.
    - transform_target (bool): Whether to apply transformation to the target variable.
    - transform_target_method (str): The method used for transforming the target variable.
    - polynomial_features (bool): Whether to create polynomial features based on numeric features.
    - polynomial_degree (int): Degree of polynomial features to be created.
    - group_features (list): List of features to be grouped for statistical feature extraction.
    - drop_groups (bool): Whether to drop the original features in the group.
    - bin_numeric_features (list): List of numeric features to be binned into intervals.
    - rare_to_value (float or None): Minimum fraction of category occurrences in a categorical column to be considered rare.
    - rare_value (str): Value to replace rare categories with.
    - feature_selection (bool): Whether to perform feature selection.
    - feature_selection_method (str): Method for feature selection.
    - feature_selection_estimator (str or sklearn estimator): Estimator used for determining feature importance.
    - n_features_to_select (int or float): Maximum number of features to select.
    - remove_multicollinearity (bool): Whether to remove multicollinear features.
    - multicollinearity_threshold (float): Threshold for identifying multicollinear features.
    - pca (bool): Whether to apply PCA for dimensionality reduction.
    - pca_method (str): Method for PCA.
    - pca_components (int, float, str, or None): Number of components to keep for PCA.
    - low_variance_threshold (float or None): Threshold for removing low variance features.
    - use_gpu (bool): Whether to use GPU for processing.

    Methods:
    - calculate_variable_info: Calculate information about variables in the given DataFrame.
    - datypess: Construct a data dictionary with the correct data types provided by the user.
    - determine_model_type: Determine the type of model based on the target variable and feature types.
    - processs_missing: Process missing values by allowing the user to configure imputation settings.
    - processs_outliers: Process outliers in the dataset and configure outlier removal settings.
    - fix_imbalances: Configure settings to fix class imbalance in the dataset.
    - normalizer: Configures normalization settings for the feature space transformation.
    - transformations: Configures transformation settings for the data to make it more normal/Gaussian-like.
    - target_transformation: Configure target transformation settings.
    - feature_engineering: Configure feature engineering settings.
    - feature_select: Configure feature selection settings.
    - setup_pycaret: Setup PyCaret for classification or regression based on the provided data and settings.
    - model_comparisons: Compare models to identify the best performing one.
    - tuned_model_results: Tune the best performing model.
    - prediction: Generate predictions using the specified model and dataset.
    - graph_results: Plot visualizations of the tuned model's performance.
    - plot_histogram: Plot a histogram for a specified column in the dataset.
    - plot_density: Plot a density plot for a specified column in the dataset.
    - plot_boxplot: Plot a boxplot for a specified column in the dataset.
    - plot_correlation: Plot a correlation heatmap for the dataset.
    - plot_model_learning: Plot learning curves for the model.
    - plot_model_validation: Plot validation curves for the model.
    - plot_model_confusion_matrix: Plot a confusion matrix for classification models.
    - plot_model_ROC: Plot the ROC curve for classification models.
    - plot_model_PR: Plot the precision-recall curve for classification models.
    - plot_model_feature_importance: Plot feature importance for tree-based models.
    """
    def __init__(self,data,target,datatypes='No',numeric_features=[],categorical_features=[],date_features=[],text_features=[],ordinal_features=[],ignore_features=[],keep_features=[],imputation_type='simple',numeric_imputation='drop',custom_numeric=5,categorical_imputation='drop',custom_categorical="",iterative_imputation_iters=5,numeric_iterative_imputer=None,categorical_iterative_imputer=None,remov_outliers='Yes',outliers_method="iforest",thresh=0.05,fix_imbalance=False, fix_imbalance_method=None,normalize=False,normalize_method='zscore',transformation=False, transformation_method='yeo-johnson',transform_target=False, transform_target_method='yeo-johnson',polynomial_features=False, polynomial_degree=2, group_features=None, drop_groups=False, bin_numeric_features=None, rare_to_value=None, rare_value='rare',feature_selection=False, feature_selection_method='classic', feature_selection_estimator='lightgbm', n_features_to_select=0.2, remove_multicollinearity=False, multicollinearity_threshold=0.9, pca=False, pca_method='linear', pca_components=None, low_variance_threshold=None,use_gpu=False):
        self.original_data = data
        self.target = target
        self.use_gpu=use_gpu
        self.missing_values_target = self.process_target()[1]
        self.data =self.process_target()[0]
        self.datatypes=datatypes
        self.numeric_features=numeric_features
        self.categorical_features=categorical_features
        self.date_features=date_features
        self.text_features=text_features
        self.ordinal_features=ordinal_features
        self.ignore_features=ignore_features
        self.keep_features=keep_features
        self.dtyps=self.datypess()           # specified dtypes
        self.model_type=self.determine_model_type()
        self.imputation_type=imputation_type
        self.numeric_imputation=numeric_imputation
        self.custom_numeric=custom_numeric
        self.categorical_imputation=categorical_imputation
        self.custom_categorical=custom_categorical
        self.iterative_imputation_iters=iterative_imputation_iters
        self.numeric_iterative_imputer=numeric_iterative_imputer
        self.categorical_iterative_imputer=categorical_iterative_imputer
        self.remov_outliers=remov_outliers
        self.outliers_method=outliers_method
        self.thresh=thresh
        self.fix_imbalance=fix_imbalance
        self.fix_imbalance_method=fix_imbalance_method
        self.normalize=normalize
        self.normalize_method=normalize_method
        self.transformation=transformation
        self.transformation_method=transformation_method
        self.transform_target=transform_target
        self.transform_target_method=transform_target_method
        self.polynomial_features=polynomial_features
        self.polynomial_degree=polynomial_degree
        self.group_features=group_features
        self.drop_groups=drop_groups
        self.bin_numeric_features=bin_numeric_features
        self.rare_to_value=rare_to_value
        self.rare_value=rare_value
        self.feature_selection=feature_selection
        self.feature_selection_method=feature_selection_method
        self.feature_selection_estimator=feature_selection_estimator
        self.n_features_to_select=n_features_to_select
        self.remove_multicollinearity=remove_multicollinearity
        self.multicollinearity_threshold=multicollinearity_threshold
        self.pca=pca
        self.pca_method=pca_method
        self.pca_components=pca_components
        self.low_variance_threshold=low_variance_threshold
        self.model_configs=self.setup_pycaret()[0]
        self.model_configs_info=self.setup_pycaret()[1]
        self.best_model=self.model_comparisons()[0]
        self.model_comparisons=self.model_comparisons()[1]
        self.tuned_model=self.tuned_model_results()[0]
        self.tuned_model_performance=self.tuned_model_results()[1]
 
    def process_target(self, **kwargs):
        """
        process target variable to remove cases with missing values.

        Parameters:
        - file_path (str): The path to the data file.
        - **kwargs: Additional keyword arguments to pass to the appropriate read function.

        Returns:
        pandas.DataFrame: The loaded DataFrame.
        """
        original_data=self.original_data
        try:
            missing_values_target = original_data[self.target].isnull().sum()
            if missing_values_target > 0:
               original_data.dropna(subset=[self.target], inplace=True)
            return [original_data,missing_values_target]
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
          
    def calculate_variable_info(self):
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
        """
        data = self.data
        missing_values_target=self.missing_values_target
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
        if missing_values_target > 0:
           print('Removed {} cases which has no target values'.format(missing_values_target))
        return variable_info
    def datypess(self):
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
        if self.datatypes=='No':
           dtypz={
             'numeric_features':self.numeric_features,
             'categorical_features':self.categorical_features,
             'date_features' :self.date_features,
             'text_features':self.text_features,
             'ordinal_features':self.ordinal_features,
             'ignore_features':self.ignore_features,
             'keep_features':self.keep_features
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
    def determine_model_type(self):
        """
        Determine the type of model based on the target variable and data.
        Returns:
        str: The type of model, either 'classification' or 'regression'.
        """
        if self.target in  (self.dtyps['categorical_features'] or []) or self.data[self.target].dtype == 'object':
           mod='classification'
        else:
          mod='regression'
        return mod
    def processs_missing(self):
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
        dtyps=self.dtyps
        data=self.data
        target=self.target
        ignored_features = dtyps.get('ignore_features', [])
        selected_columns = [col for col in data.columns if col not in (ignored_features or []) and col != target]
        missing= data[selected_columns].isnull().sum()
        if missing.any() > 0:
           imputes={
             'imputation_type':self.imputation_type,
             'numeric_imputation':self.numeric_imputation,
             'categorical_imputation':self.categorical_imputation,
             'iterative_imputation_iters':self.iterative_imputation_iters,
             'numeric_iterative_imputer':self.numeric_iterative_imputer,
             'categorical_iterative_imputer':self.categorical_iterative_imputer
             }
           if self.numeric_imputation == 'custom':
              imputes['numeric_imputation'] = self.custom_numeric

           # Conditionally update the dictionary if categorical_imputation is 'custom'
           if self.categorical_imputation == 'custom':
              imputes['categorical_imputation'] = self.custom_categorical
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
    def processs_outliers(self):
        """
        Process outliers in the dataset and configure outlier removal settings.
        Parameters:
        - remov_outliers (str): Indicates whether to remove outliers or not. Options are 'Yes' or 'No'. Default is 'False'.
        - outliers_method (str): The method for outlier detection. Default is "iforest".
        - thresh (float): The threshold for outlier detection. Default is 0.05.
 
        Returns:
        dict: A dictionary containing the configured outlier removal settings.
        """
        remov_outliers=self.remov_outliers
        outliers_method=self.outliers_method
        thresh=self.thresh
        
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
    def fix_imbalances(self):
        """
        Configure settings to fix class imbalance in the dataset.

        Parameters:
        - fix_imbalance (bool): Indicates whether to fix class imbalance or not. Default is False.
        - fix_imbalance_method (str): The method to fix class imbalance. Default is None.

        Returns:
        dict: A dictionary containing the configured settings to fix class imbalance.
        """
        fix_imbalance=self.fix_imbalance
        fix_imbalance_method=self.fix_imbalance_method
        out = {
          'fix_imbalance': fix_imbalance,
          'fix_imbalance_method': fix_imbalance_method
          }
        return out
    def normalizer(self):
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
        normalize=self.normalize
        normalize_method=self.normalize_method
        out={
          'normalize':normalize,
          'normalize_method':normalize_method
          }
        return out
    def transformations(self):
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
        transformation=self.transformation
        transformation_method=self.transformation_method
        out = {
          'transformation': transformation,
          'transformation_method': transformation_method
          }
        return out
    def target_transformation(self):
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
        data=self.data
        target=self.target
        datatypes=self.dtyps
        transform_target=self.transform_target
        transform_target_method=self.transform_target_method
        out = {
          'transform_target': transform_target,
          'transform_target_method': transform_target_method
          }
        return out
    def feature_engineering(self):
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
        polynomial_features=self.polynomial_features
        polynomial_degree=self.polynomial_degree
        group_features=self.group_features
        drop_groups=self.drop_groups
        bin_numeric_features=self.bin_numeric_features
        rare_to_value=self.rare_to_value
        rare_value=self.rare_value
        
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
    def feature_select(self):
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
        feature_selection=self.feature_selection
        feature_selection_method=self.feature_selection_method
        feature_selection_estimator=self.feature_selection_estimator
        n_features_to_select=self.n_features_to_select
        remove_multicollinearity=self.remove_multicollinearity
        multicollinearity_threshold=self.multicollinearity_threshold
        pca=self.pca
        pca_method=self.pca_method
        pca_components=self.pca_components
        low_variance_threshold=self.low_variance_threshold
        
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
    def setup_pycaret(self):
        """
        Setup PyCaret for classification or regression based on the provided data and settings.

        Parameters:
        - data (DataFrame): The input dataset.
        - target (str): The target variable.
        - *args (dict): Additional dictionaries containing settings for PyCaret setup.

        Returns:
        object: PyCaret setup object for classification or regression.
        """
        data=self.data
        target=self.target
        
        # Combine data and target into a dictionary
        targetanddatadict = {'data': data, 'target': target,'use_gpu':self.use_gpu}
        
        args=[self.dtyps,self.processs_missing(),self.fix_imbalances(),self.processs_outliers(),self.normalizer(),self.transformations(),self.target_transformation(),self.feature_engineering(),self.feature_select()]

          # Combine all dictionaries into a single master dictionary
        master_dict = targetanddatadict.copy()
        for d in args:
            master_dict.update(d)
    
        # Setup PyCaret based on the provided dictionaries
        if self.model_type=='classification':
           del master_dict['transform_target']
           del master_dict['transform_target_method']
           settings = classification.setup(**master_dict)
           settingsinfo = classification.pull(settings)
        else:
          del master_dict['fix_imbalance']
          del master_dict['fix_imbalance_method']
          settings = regression.setup(**master_dict)
          settingsinfo = regression.pull(settings)
        return [settings,settingsinfo]
    def model_comparisons(self):
        """
        Compare different models and select the best one based on the dataset.

        Returns:
        object: The best model selected based on the comparisons.
        """
        if self.model_type=='classification':
           best_model=classification.compare_models()
           comparisons=classification.pull(best_model)
        else:
           best_model=regression.compare_models()
           comparisons=regression.pull(best_model) 
        return [best_model,comparisons]
    def tuned_model_results(self):
        """
        Tune the selected model and display the results.
        
        Returns:
        object: The tuned model with improved performance.
        """
        if self.model_type=='classification':
           cat = classification.create_model(self.best_model)
           cat_tune = classification.tune_model(cat)
           tunedresults=classification.pull(cat_tune)
        else:
           cat = regression.create_model(self.best_model)
           cat_tune = regression.tune_model(cat)
           tunedresults=regression.pull(cat_tune)
        return [cat_tune,tunedresults]
    def prediction(self,newdata):
        """
        Generate predictions for new data using the tuned model.

        Parameters:
        - newdata (pd.DataFrame): prediction data.

        Returns:
        DataFrame: A DataFrame containing the predictions.
        """
        try:
            # Strip column names
            newdata.rename(columns=lambda x: x.strip(), inplace=True)
            if self.target in newdata.columns:
               newdata.rename(columns={self.target:self.target+'_zipper'}, inplace=True)
            if self.model_type=='classification':
               pred=classification.predict_model(self.tuned_model,data=newdata)
               pred.rename(columns={self.target+'_zipper':self.target}, inplace=True)
            else:
              pred=regression.predict_model(self.tuned_model,data=newdata)
              pred.rename(columns={self.target+'_zipper':self.target}, inplace=True)
            return pred
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    def graph_results(self):
        """
        Visualize the results of the tuned model.
        Returns:
        matplotlib.pyplot.figure: The matplotlib figure object containing the visualization.
        """
        if self.model_type=='classification':
           return classification.plot_model(self.tuned_model)
        else:
          return regression.plot_model(self.tuned_model)
    def plot_histogram(self,bins=50):
        """
        Plot a histogram for a specified column in the dataset.
        
        Parameters:
        - data (DataFrame): The input dataset.
        - column (str): The column to plot the histogram for.
        - bins (int, optional): The number of bins for the histogram. Defaults to 50.
        
        Returns:
        matplotlib.pyplot.figure: The matplotlib figure object containing the histogram plot.
        """
        data=self.data
        column=self.target
        plt.figure(figsize=(15, 10))
        sns.histplot(data=self.data,x=column, bins=bins, cbar=True)
        return plt  
    def plot_countplot(self):
        """
        Plot a countplot for the specified column in the dataset.

        Parameters:
        - data (DataFrame): The input dataset.
        - x (str): The column to plot on the x-axis.
        - hue (str, optional): The column to differentiate by color.
        Returns:
        matplotlib.pyplot.figure: The matplotlib figure object containing the countplot.
        """
        data=self.data
        target=self.target
        plt.figure(figsize=(10, 6))
        sns.countplot(data=self.data, x=self.target, hue=self.target)
        return plt
    def plot_correlation_matrix(self):
        """
        Plot a correlation matrix heatmap for numeric columns in the dataset.

        Parameters:
        - data (DataFrame): The input dataset.
        - dtyps (dict): A dictionary containing information about data types.
        
        Returns:
        matplotlib.pyplot.figure: The matplotlib figure object containing the correlation matrix heatmap.
        """
        data=self.data
        dtyps=self.dtyps
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
    def data_head(self,n=5):
        """
        Display the first n rows of the dataset.
        
        Parameters:
        - n (int, optional): The number of rows to display. Defaults to 5.
        Returns:
        DataFrame: A DataFrame containing the first n rows of the dataset.
        """
        return data.head(n)

