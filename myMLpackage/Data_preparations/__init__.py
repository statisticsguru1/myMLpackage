"""
Data Preparation Module
=======================

This module provides functions for data preparation tasks such as handling data types, processing the target variable, and dealing with missing values and outliers.

Functions:
    datypess(datatypes='No',numeric_features=[],categorical_features=[],date_features=[],text_features=[],ordinal_features=[],ignore_features=[],keep_features=[]):
        Constructs a data dictionary with the correct data types provided by the user.
    determine_model_type(data, target, dtyps):
        Determine the type of model based on the target variable and data types.
    process_target(data, target):
        Process the target variable by handling missing values.
    processs_missing(missing, imputation_type=None, numeric_imputation='drop', custom_numeric=5, categorical_imputation='drop', custom_categorical="", iterative_imputation_iters=5, numeric_iterative_imputer=None, categorical_iterative_imputer=None):
        Process missing values by allowing the user to configure imputation settings.
    processs_outliers(remov_outliers='Yes', outliers_method="iforest", thresh=0.05):
        Process outliers in the dataset and configure outlier removal settings.
    fix_imbalances(fix_imbalance=False, fix_imbalance_method=None):
        Configure settings to fix class imbalance in the dataset.

Usage Example:
--------------
Import the module:

    import myMLpackage.Data_preparations as dp

Construct a data dictionary with specified data types:

    dtyps = dp.datypess(numeric_features=['Age', 'Fare'], categorical_features=['Sex', 'Embarked'])

Determine the type of model based on the target variable and data types:

    model_type = dp.determine_model_type(data, 'Survived', dtyps)

Process the target variable:

    processed_data = dp.process_target(data, 'Survived')

Process missing values:

    missing_values = data.isnull().sum()
    imputation_settings = dp.processs_missing(missing_values)

Process outliers:

    outlier_settings = dp.processs_outliers()

Fix class imbalance:

    imbalance_settings = dp.fix_imbalances(fix_imbalance=True, fix_imbalance_method='SMOTE')

"""
from .Data_preparations import *