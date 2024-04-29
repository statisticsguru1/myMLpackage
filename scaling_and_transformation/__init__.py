"""
Scaling and Transformation Module
=================================

This module provides functions for scaling and transforming features in the dataset.

Functions:
    normalizer(normalize=False, normalize_method='zscore'):
        Configures normalization settings for the feature space transformation.
    transformations(transformation=False, transformation_method='yeo-johnson'):
        Configures transformation settings for the data to make it more normal/Gaussian-like.
    target_transformation(data, target, datatypes, transform_target=False, transform_target_method='yeo-johnson'):
        Configure target transformation settings.

Usage Example:
--------------
Import the module:

    import myMLpackage.scaling_and_transformation as st

Configure normalization settings:

    normalization_settings = st.normalizer(normalize=True, normalize_method='minmax')

Configure transformation settings:

    transformation_settings = st.transformations(transformation=True, transformation_method='quantile')

Configure target transformation settings:

    target_transformation_settings = st.target_transformation(data, 'SalePrice', datatypes, transform_target=True, transform_target_method='yeo-johnson')

"""

from .scaling_and_transformation import *