"""
Feature Selection Module
========================

This module provides functions for configuring feature selection settings for pycaret classification or regression setup.

Functions:
    feature_select(feature_selection=False, feature_selection_method='classic', feature_selection_estimator='lightgbm', n_features_to_select=0.2, remove_multicollinearity=False, multicollinearity_threshold=0.9, pca=False, pca_method='linear', pca_components=None, low_variance_threshold=None):
        Configure feature selection settings.

Usage Example:
--------------
Import the module:

    import myMLpackage.feature_selection as fs

Configure feature selection settings:

    feature_selection_settings = fs.feature_select(feature_selection=True, feature_selection_method='classic', feature_selection_estimator='randomforest', n_features_to_select=0.3, remove_multicollinearity=True, multicollinearity_threshold=0.8, pca=True, pca_method='kernel', pca_components=10, low_variance_threshold=0.01)

"""

from .feature_selection import *