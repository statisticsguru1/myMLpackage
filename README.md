# MyMLpackage

MyMLpackage is a Python package that provides dedicated functions for various operations and settings related to machine learning training. The package consists the following modules:

- **General utility** General Function to support other functions in graphing etc
- **Data Preparation**: Functions to prepare data for machine learning tasks.
- **Data Transformation and Scaling**: Functions for transforming and scaling data.
- **Feature Engineering**: Functions for feature engineering tasks.
- **Feature Selection**: Functions to select relevant features for modeling.
- **Modelling**: Functions for building machine learning models.
- **Prediction** Functions to make predictions using the trained model
- **Automation class** A python class which can be initialized with data and inputs to automate ML training.
- **Data** Inbuilt datasets.
- **streamlit wraparounds** Provides wrap around functions to add streamlit inputs functionality. 

most of these modules bundle the inputs into a master data dictionary, which is then passed to the Pycaret setup function for further processing.

## Installation

You can install MyMLpackage from `pip install git+https://github.com/wfa19/myMLpackage.git`

```bash
 pip install git+https://github.com/wfa19/myMLpackage.git
```
## Usage
```bash
import myMLpackage                                               
from myMLpackage import general_utility as gu
from myMLpackage import Data_preparations as dp
from myMLpackage import automation_class as ac

data=gu.load_experimental_dataset('heart')
data['target'] = data['target'].replace({1: 'Heart disease', 0: 'No Heart disease'})
data.head()
```
## Print data info
```bash
data.info()
```
## Data preparation settings 
### Setting up data types 
```bash
 datypess()
```
### processing missing cases
```bash
processs_missing(missing)
```
### Set up outlier processing 
```bash
processs_outliers(remov_outliers='Yes',outliers_method="iforest",thresh=0.05)
```
see documentation about the other modules

```bash
help(myMLpackage)  # Package documentation
help(myMLpackage.general_utility) # general utility module documentation
help(myMLpackage.Data_preparations) # data preparation module documentation
```
## Automated Class
MyMLpackage features an automated class that initializes with data and inputs for training, tuning, evaluating, and visualizing ML models. This class streamlines the machine learning workflow and provides convenient methods for managing the entire process.

### using automation class
Minimal initialization
```bash
model=ac.Modelling(data,'target')
```

### get model configuration
```bash
model.model_configs_info
```
### see data information
```bash
model.calculate_variable_info()
```
### See trained model type
```bash
model.determine_model_type()
```
### Graph categorical target
```bash
model.plot_countplot()
```
### Graph correlation matrix of features
```bash
model.plot_correlation_matrix()
```
### model comparison
```bash
model.model_comparisons
model.tuned_model             # best model after tuning
model.tuned_model_performance # performance of tuned model
```
### graph best model
```bash
model.graph_results()
```
### Predict new data
```bash
model.prediction(newdata)
```

### see more methods of the class

```bash
help(model)
```
## Streamlit app
MyMLpackage also includes a Streamlit app for providing a user-friendly interface where users can load data and configure ML training settings for Pycaret. The Streamlit app can be found in the [streamlitapp](https://github.com/wfa19/streamlitapp)
 repository. You can run the app directly [here](https://appapp-qnebc74f2xhhqkx2s6ypwr.streamlit.app/) 
## License
This project is licensed under the MIT License.

```bash
Feel free to adjust the formatting and content to fit your preferences and requirements. If you have any further questions or need assistance, feel free to ask!
```

