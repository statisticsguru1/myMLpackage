"""
Feature Engineering Module
==========================

This module provides functions for configuring feature engineering settings, which can be used in pycaret set up.

Functions:
    feature_engineering(polynomial_features=False, polynomial_degree=2, group_features=None, drop_groups=False, bin_numeric_features=None, rare_to_value=None, rare_value='rare'):
        Configure feature engineering settings.

Usage Example:
--------------
Import the module:

    import myMLpackage.feature_engineering as fe

Configure feature engineering settings:

    feature_engineering_settings = fe.feature_engineering(polynomial_features=True, polynomial_degree=3, group_features=['feature1', 'feature2'], drop_groups=True, bin_numeric_features=['numeric_feature1', 'numeric_feature2'], rare_to_value=0.05, rare_value='other')

"""

from .feature_engineering import *