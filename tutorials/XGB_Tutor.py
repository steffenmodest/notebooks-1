# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'xgboost'))
	print(os.getcwd())
except:
	pass
#%%
from IPython import get_ipython

#%% [markdown]
# # Introduction to XGBoost with RAPIDS
# #### By Paul Hendricks
# -------
# 
# While the world’s data doubles each year, CPU computing has hit a brick wall with the end of Moore’s law. For the same reasons, scientific computing and deep learning has turned to NVIDIA GPU acceleration, data analytics and machine learning where GPU acceleration is ideal. 
# 
# NVIDIA created RAPIDS – an open-source data analytics and machine learning acceleration platform that leverages GPUs to accelerate computations. RAPIDS is based on Python, has pandas-like and Scikit-Learn-like interfaces, is built on Apache Arrow in-memory data format, and can scale from 1 to multi-GPU to multi-nodes. RAPIDS integrates easily into the world’s most popular data science Python-based workflows. RAPIDS accelerates data science end-to-end – from data prep, to machine learning, to deep learning. And through Arrow, Spark users can easily move data into the RAPIDS platform for acceleration.
# 
# In this notebook, we'll show the acceleration one can gain by using GPUs with XGBoost in RAPIDS.
# 
# **Table of Contents**
# 
# * Setup
# * Load Libraries
# * Load/Simulate Data
#   * Load Data
#   * Simulate Data
#   * Split Data
#   * Check Dimensions
# * Convert NumPy data to DMatrix format
# * Set Parameters
# * Train Model
# * Conclusion
#%% [markdown]
# ## Setup
# 
# This notebook was tested using the `nvcr.io/nvidia/rapidsai/rapidsai:0.5-cuda10.0-runtime-ubuntu18.04-gcc7-py3.7` Docker container from [NVIDIA GPU Cloud](https://ngc.nvidia.com) and run on the NVIDIA Tesla V100 GPU. Please be aware that your system may be different and you may need to modify the code or install packages to run the below examples. 
# 
# If you think you have found a bug or an error, please file an issue here: https://github.com/rapidsai/notebooks/issues
# 
# To start, let's see what hardware we're working with.

#%%
get_ipython().system('nvidia-smi')

#%% [markdown]
# Next, let's see what CUDA version we have.

#%%
get_ipython().system('nvcc --version')

#%% [markdown]
# ## Load Libraries
# 
# Let's load some of the libraries within the RAPIDs ecosystem and see which versions we have.

#%%
import numpy as np; print('numpy Version:', np.__version__)
import pandas as pd; print('pandas Version:', pd.__version__)
import xgboost as xgb; print('XGBoost Version:', xgb.__version__)

#%% [markdown]
# ## Load/Simulate data
# 
# ### Load Data
# 
# We can load the data using `pandas.read_csv`.
# 
# ### Simulate Data
# 
# Alternatively, we can simulate data for our train and validation datasets. The features will be tabular with `n_rows` and `n_columns` in the training dataset, where each value is either of type `np.float32` if the data is numerical or `np.uint8` if the data is categorical. Both numerical and categorical data can also be combined; for this experiment, we have ignored this combination.

#%%
# helper function for simulating data
def simulate_data(m, n, k=2, numerical=False):
    if numerical:
        features = np.random.rand(m, n)
    else:
        features = np.random.randint(2, size=(m, n))
    labels = np.random.randint(k, size=m)
    return np.c_[labels, features].astype(np.float32)


# helper function for loading data
def load_data(filename, n_rows):
    if n_rows >= 1e9:
        df = pd.read_csv(filename)
    else:
        df = pd.read_csv(filename, nrows=n_rows)
    return df.values.astype(np.float32)


#%%
# settings
LOAD = False
n_rows = int(1e5)
n_columns = int(100)
n_categories = 2


#%%
get_ipython().run_cell_magic('time', '', "\nif LOAD:\n    dataset = load_data('/tmp', n_rows)\nelse:\n    dataset = simulate_data(n_rows, n_columns, n_categories)\nprint(dataset.shape)")

#%% [markdown]
# ### Split Data
# 
# We'll split our dataset into a 80% training dataset and a 20% validation dataset.

#%%
# identify shape and indices
n_rows, n_columns = dataset.shape
train_size = 0.80
train_index = int(n_rows * train_size)

# split X, y
X, y = dataset[:, 1:], dataset[:, 0]
del dataset

# split train data
X_train, y_train = X[:train_index, :], y[:train_index]

# split validation data
X_validation, y_validation = X[train_index:, :], y[train_index:]

#%% [markdown]
# ### Check Dimensions
# 
# We can check the dimensions and proportions of our training and validation dataets.

#%%
# check dimensions
print('X_train: ', X_train.shape, X_train.dtype, 'y_train: ', y_train.shape, y_train.dtype)
print('X_validation', X_validation.shape, X_validation.dtype, 'y_validation: ', y_validation.shape, y_validation.dtype)

# check the proportions
total = X_train.shape[0] + X_validation.shape[0]
print('X_train proportion:', X_train.shape[0] / total)
print('X_validation proportion:', X_validation.shape[0] / total)

#%% [markdown]
# ## Convert NumPy data to DMatrix format
# 
# With out data simulated and formatted as NumPy arrays, our next step is to convert this to a `DMatrix` object that XGBoost can work with. We can instantiate an object of the `xgboost.DMatrix` by passing in the feature matrix as the first argument followed by the label vector using the `label=` keyword argument. To learn more about XGBoost's support for data structures other than NumPy arrays, see the documentation for the Data Interface:
# 
# 
# https://xgboost.readthedocs.io/en/latest/python/python_intro.html#data-interface
# 

#%%
get_ipython().run_cell_magic('time', '', '\ndtrain = xgb.DMatrix(X_train, label=y_train)\ndvalidation = xgb.DMatrix(X_validation, label=y_validation)')

#%% [markdown]
# ## Set Parameters
# 
# There are a number of parameters that can be set before XGBoost can be run. 
# 
# * General parameters relate to which booster we are using to do boosting, commonly tree or linear model
# * Booster parameters depend on which booster you have chosen
# * Learning task parameters decide on the learning scenario. For example, regression tasks may use different parameters with ranking tasks.
# 
# For more information on the configurable parameters within the XGBoost module, see the documentation here:
# 
# 
# https://xgboost.readthedocs.io/en/latest/parameter.html

#%%
# instantiate params
params = {}

# general params
general_params = {'silent': 1}
params.update(general_params)

# booster params
n_gpus = 1
booster_params = {}

if n_gpus != 0:
    booster_params['tree_method'] = 'gpu_hist'
    booster_params['n_gpus'] = n_gpus
params.update(booster_params)

# learning task params
learning_task_params = {'eval_metric': 'auc', 'objective': 'binary:logistic'}
params.update(learning_task_params)
print(params)

#%% [markdown]
# ## Train Model
# 
# Now it's time to train our model! We can use the `xgb.train` function and pass in the parameters, training dataset, the number of boosting iterations, and the list of items to be evaluated during training. For more information on the parameters that can be passed into `xgb.train`, check out the documentation:
# 
# 
# https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.train

#%%
# model training settings
evallist = [(dvalidation, 'validation'), (dtrain, 'train')]
num_round = 10


#%%
get_ipython().run_cell_magic('time', '', '\nbst = xgb.train(params, dtrain, num_round, evallist)')

#%% [markdown]
# ## Conclusion
# 
# To learn more about RAPIDS, be sure to check out: 
# 
# * [Open Source Website](http://rapids.ai)
# * [GitHub](https://github.com/rapidsai/)
# * [Press Release](https://nvidianews.nvidia.com/news/nvidia-introduces-rapids-open-source-gpu-acceleration-platform-for-large-scale-data-analytics-and-machine-learning)
# * [NVIDIA Blog](https://blogs.nvidia.com/blog/2018/10/10/rapids-data-science-open-source-community/)
# * [Developer Blog](https://devblogs.nvidia.com/gpu-accelerated-analytics-rapids/)
# * [NVIDIA Data Science Webpage](https://www.nvidia.com/en-us/deep-learning-ai/solutions/data-science/)

#%%



