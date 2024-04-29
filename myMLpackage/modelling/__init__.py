"""
Module: modelling

Description:
This module provides functions for setting up PyCaret for classification or regression based on the provided data 
and settings. It includes functions for determining the model type, setting up PyCaret, comparing different models, 
tuning the selected model, and plotting visualizations of the tuned model's performance.

Functions:
- model_typ: Determine the model type based on the target variable and data types.
- setup_pycaret: Setup PyCaret for classification or regression based on the provided data and settings.
- model_comparisons: Compare different models and select the best one based on the specified model type.
- tuned_model: Tune the selected model based on the specified model type.
- graph_results: Plot visualizations of the tuned model's performance.

Usage:
Import the module and use the provided functions to set up and tune machine learning models using PyCaret.

Example:
```python
import model_setup

# Determine the model type
model_type = model_setup.model_typ(data, target, dtyps)

# Setup PyCaret for classification or regression
pycaret_setup = model_setup.setup_pycaret(data, target, use_gpu=True, **other_settings)

# Compare different models and select the best one
best_model = model_setup.model_comparisons(model_type)

# Tune the selected model
tuned_model = model_setup.tuned_model(best_model, model_type)

# Plot visualizations of the tuned model's performance
model_setup.graph_results(tuned_model, model_type)
"""
from .modelling import *