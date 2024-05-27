# What are the assumptions of logistic regression in relation to continuous values.
# How to check the assumptions of logistic regression.


import pandas as pd

df = pd.read_csv("social_media_train.csv", index_col=[0])
df.head()
features_cont = ['ratio_numlen_username', 'len_fullname', 'ratio_numlen_fullname',
                'len_desc', 'num_posts', 'num_followers', 'num_following']


#  like linear regression, logistic regression makes a number of assumptions. The following are relevant for continuous data:

# The features should not correlate strongly with each other.
# There should be a linear relationship between the features and the sigmoid-transformed probabilities.


# This assumption is shared by logistic regression and linear regression
# we'll calculate a correlation matrix for the continuous features here as well

corr = df.corr()

import seaborn as sns

sns.heatmap(corr.abs()) # Heat map of absolute values of correlation matrix

# Then you have the following options to deal with collinearity:
#   Use regularization to weight the columns differently.
#   Use PCA to extract the most important features.
#   Use domain knowledge to either choose only one of the two columns or create a new feature from both columns and discard the original columns.


# **Important:** 
# Just like a linear regression, logistic regression is not robust against outliers in the data. 
# This is one way in which it differs from k-Nearest Neighbors. 
# The k-Nearest-Neighbors classification method is extremely robust, since only the local neighborhood is used for classification. 
# Outliers are most likely extreme values outside the neighborhood and are ignored as a result.


# Remember:
# Logistic regression assumes that continuous features are not strongly correlated.
# A correlation matrix shows the correlations between continuous features.
