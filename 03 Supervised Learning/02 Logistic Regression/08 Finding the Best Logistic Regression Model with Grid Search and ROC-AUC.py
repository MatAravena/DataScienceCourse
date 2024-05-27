# Preparing data

# module import
import pandas as pd
import pdpipe as pdp

# label encoding dictionary
dict_label_encoding = {'Yes': 1, 'No': 0}

# will save DataFrames
df_list = []

for df_str in ['social_media_train.csv', 
               'social_media_test.csv',
               'social_media_aim.csv']:
    
    # data read in
    df = pd.read_csv(df_str, index_col=[0])

    # label encoding    
    df.loc[:, 'profile_pic'] = df.loc[:, 'profile_pic'].replace(dict_label_encoding)
    df.loc[:, 'extern_url'] = df.loc[:, 'extern_url'].replace(dict_label_encoding)
    df.loc[:, 'private'] = df.loc[:, 'private'].replace(dict_label_encoding)

    # append to list
    df_list.append(df)


# creating data sets
df_train = df_list[0]
df_test = df_list[1]
df_aim = df_list[2]

# one-hot encoding
onehot = pdp.OneHotEncode(["sim_name_username"], drop_first=False)
df_train = onehot.fit_transform(df_train) # only ever fit to training set!
df_test = onehot.transform(df_test)
df_aim = onehot.transform(df_aim)

# look at data
df_train.head()



# Grid search

# So far we've evaluated two models to find the better one. However, in the lesson Grid Search 
# you learned that Python can do that for us as well. Using a grid search automatically finds the optimal hyperparameters

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

features_train = df_train.iloc[:,1:]
target_train =  df_train.iloc[:,0]

# Important: Not all solver settings support all options. Some also take longer or need more memory. You can find more on the documentation page.

# Since we want to try out the regularization of both ridge regression and the LASSO regression in this notebook, 
# we can't use 'lbfgs' as before. 'saga' is the only setting that supports both types of regularization.

# For the algorithm to find a good result, it has to make quite a lot of attempts. 
# So you should increase max_iter from the standard 100 iterations. 10000 iterations (1e4) should be enough.

pipeline_log = Pipeline([('scaler', StandardScaler()),
                         ('classifier', LogisticRegression(solver='saga',
                                                           max_iter=1e4,
                                                           random_state=42))])

# penalty: whether the regularization proceeds like with Ridge ('l2') or like with LASSO ('l1')
# C: Regularization weakness, see Logistical Regression with and without Regularization

import numpy as np
np.set_printoptions(suppress=True)  # avoid scientific notation

C_values = np.geomspace(start=0.001, stop=1000, num=14)

search_space_grid = [{'classifier__penalty': ['l1', 'l2'],
                      'classifier__C': C_values}]

model_grid = GridSearchCV(estimator=pipeline_log,
                          param_grid=search_space_grid,
                          scoring='roc_auc',
                          cv=5,
                          n_jobs=-1)

# Before we do the grid search, we'll deactivate the DataConversionWarning, which would otherwise occur very often although you can safely ignore it
from sklearn.exceptions import DataConversionWarning
import warnings

warnings.filterwarnings(action='ignore', category=DataConversionWarning)

model_grid.fit(features_train, target_train)
print(model_grid.best_estimator_)
print(model_grid.best_score_ )

# Pipeline(memory=None,
#          steps=[('scaler',
#                  StandardScaler(copy=True, with_mean=True, with_std=True)),
#                 ('classifier',
#                  LogisticRegression(C=4.923882631706742, class_weight=None,
#                                     dual=False, fit_intercept=True,
#                                     intercept_scaling=1, l1_ratio=None,
#                                     max_iter=10000.0, multi_class='auto',
#                                     n_jobs=None, penalty='l2', random_state=42,
#                                     solver='saga', tol=0.0001, verbose=0,
#                                     warm_start=False))],
#          verbose=False)
# 0.9657175042242944




# Evaluating the model with test data

features_test = df_test.iloc[:, 1:]
target_test = df_test.iloc[:, 0]

target_test_pred_proba = model_grid.predict_proba(features_test)

from sklearn.metrics import roc_auc_score
roc_auc_score(target_test, target_test_pred_proba[:, 1])

# 0.9391666666666667


# The training data is guaranteed to be representative of the validation data because it comes from the same data set. However, the is not necessarily the case with the test data. The trained model therefore matches the validation data better (which is similar to the training data) than the test data (which is not necessarily as similar to the training data). This makes the model look better when evaluating with validation data than when evaluating with test data.
# A grid search optimizes the hyperparameter settings for the training data. In extreme cases, this can result in the hyperparameter settings being overfitted to the training data. This problem is usually not as pronounced as the overfitting of the model coefficients to the training data during model fitting (see The Overfitting Problem (Module 1, Chapter 1)). However, this can still result in slightly lower model performance with test data than with validation data.
# There is typically only one test data set and this is often relatively small. This could be a better or worse match to the trained model by coincidence. The calculated model quality can therefore go up or down more easily than with the validation data. For the validation data, the model quality measures are calculated at each step of the cross-validation and then averaged. This compensates for random upward or downward deviations.



# Predictions
features_aim = df_aim.copy()
df_aim.loc[:, 'fake_pred_proba'] = model_grid.predict_proba(features_aim)[:, 1]
df_aim.loc[:, 'fake_pred'] = model_grid.predict(features_aim)

# avoid scientific notation
pd.set_option('display.float_format', lambda x: '%.3f' % x)
df_aim



# Remember:

# You can output the best model and the corresponding best metric for a grid search with the sklearn attributes my_model.best_estimator_ and my_model.best_score_ (always ending with an underscore).
# With np.geomspace() you can generate a series of numbers with increasing distances between the numbers
# There can be several reasons why model performance with validation data differs slightly from model performance with test data.
