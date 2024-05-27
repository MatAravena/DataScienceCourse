# PCA
# Principal Component Analysis
# Dimensionality reduction belongs to the field of unsupervised learning. 
# So it works without a target vector. Instead, information about the structure of the data is used directly to 
# determine whether and how the data can be transformed so that it has fewer features. Different algorithms have different approaches. 
# However, the basic assumption is always that there are features that contain redundant or little information about the data. 


import pandas as pd 
df = pd.read_csv('swim_records.txt', sep='\t', parse_dates=['date'])
df.head()

import seaborn as sns

ax = sns.scatterplot(data=df, 
                     x='date',
                     y='seconds')
ax.set(title='Swimming World records over time');

df.loc[:, 'date_int'] = pd.to_numeric(df.loc[:, 'date'])

from sklearn.preprocessing import StandardScaler
arr_std = StandardScaler().fit_transform(df.loc[:, ['date_int', 'seconds']])

sns.scatterplot(x=arr_std[:,0], y=arr_std[:,1]);

from sklearn.decomposition import PCA
model = PCA(n_components=1)
model.fit(arr_std)

model.components_

arr_1d = model.transform(arr_std) 

import numpy as np
sns.scatterplot(y=0, x=arr_1d[:,0]);

model.explained_variance_rataio_




# Reducing the amount of storage space needed and increasing speed
# Reducing noise in the data
# Visualizing high-dimensional data
# Extracting relevant features from the data



# Remember:

# Standardize data before PCA
# Indicate the number of features (principal components) with PCA(n_components=1)
# Evaluate the quality of the projection with model.explained_variance_ratio_