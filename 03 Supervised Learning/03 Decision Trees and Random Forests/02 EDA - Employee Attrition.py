# The attrition data
# Los datos de deserción


import pandas as pd
df_train = pd.read_csv('attrition_train.csv')
features_train = df_train.iloc[:,1:]
target_train = df_train.iloc[:,0]
df_train.head()

attrition_prop_train = pd.crosstab(index=target_train, columns='count', normalize='columns')
attrition_prop_train

# module import
import matplotlib.pyplot as plt

# initialize figure and axes
fig, ax = plt.subplots()

# draw pie chart
attrition_prop_train.plot(kind='pie',
                          y='count',
                          labels=['Stayed at company', 'Left company'],  # set labels
                          autopct='%1.1f%%',  # print values to
                          legend=False,  # do not print legend
                          startangle=90,  # rotate pie
                          title='Attrition in training data',  # set title
                          ax=ax)

# optimize pie chart
ax.set_ylabel('')  # do not print "count"
ax.set_aspect('equal')  # draw a circle, not an ellipse



# Test data
df_test = pd.read_csv('attrition_test.csv')
features_test = df_test.iloc[:,1:]
target_test = df_test.iloc[:,0]

attrition_prop_test = pd.crosstab(index=target_test, columns='count', normalize='columns')

# initialize figure and axes
fig, ax = plt.subplots()

# draw pie chart
attrition_prop_test.plot(kind='pie',
                          y='count',
                          labels=['Stayed at company', 'Left company'],  # set labels
                          autopct='%1.1f%%',  # print values to
                          legend=False,  # do not print legend
                          startangle=90,  # rotate pie
                          title='Attrition in training data',  # set title
                          ax=ax)

# optimize pie chart
ax.set_ylabel('')  # do not print "count"
ax.set_aspect('equal')  # draw a circle, not an ellipse


df_train.describe().T
# If the dispersion is zero, the column doesn't contain any information that a machine learning model can learn.

df_train = df_train.drop(['over18','standardhours'], axis=1)
df_test = df_test.drop(['over18','standardhours'], axis=1)


# scatterplot matrix
import seaborn as sns

# draw correlogram
sns.pairplot(df_train)



# Principal Component Analysis of the "years" features

num_cols = ['age',
            'distancefromhome',
            'monthlyincome', 
            'numcompaniesworked', 
            'percentsalaryhike', 
            'trainingtimeslastyear', 
            'totalworkingyears', 
            'years_atcompany', 
            'years_currentrole', 
            'years_lastpromotion',
            'years_withmanager']

features_train.loc[:, num_cols].corr()

sns.heatmap(features_train.loc[:, num_cols].corr())


col_correlated = ['totalworkingyears', 
                  'years_atcompany',
                  'years_currentrole',
                  'years_lastpromotion',
                  'years_withmanager']

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

std_pca = Pipeline([('std', StandardScaler()), 
                    ('pca', PCA(n_components=0.8))])


# Fit and correlated the columns
arr_years_train = std_pca.fit_transform(features_train.loc[:, col_correlated])
arr_years_train.shape

# Drop columns from features that are now correlated
features_train = features_train.drop(col_correlated, axis=1)

# Add the new PCA values to the features
features_train.loc[:, 'pca_years_0'] = arr_years_train[:, 0]
features_train.loc[:, 'pca_years_1'] = arr_years_train[:, 1]


# Same within Test DF
# pca
arr_years_test = std_pca.transform(features_test.loc[:, col_correlated])

# remove old features
features_test = features_test.drop(col_correlated, axis=1)

# add pca features
features_test.loc[:, 'pca_years_0'] = arr_years_test[:, 0]
features_test.loc[:, 'pca_years_1'] = arr_years_test[:, 1]


# Do the new columns 'pca_years_0' and 'pca_years_1' correlate with each other in the training data or in the test data? They shouldn't.
# Do the new columns 'pca_years_0' and 'pca_years_1' correlate with other continuous columns very strongly in the training data or in the test data? They shouldn't.
# Is the the new 'pca_years_0' column on a similar scale in training data and test data? This should be the case.
# Is the scale of the new column 'pca_years_1' similar in training data and test data? This should be the case.


print(features_train.shape)
features_train.head()


# Remmber 
# You can generally remove features that only have one value.
# The training data should be similar to the test data.
# If features are correlated, principal component analysis is a useful tool for reducing the number of features.

