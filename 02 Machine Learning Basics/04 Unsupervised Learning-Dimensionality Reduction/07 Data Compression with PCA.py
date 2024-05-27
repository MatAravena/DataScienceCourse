import sqlalchemy as sa
engine = sa.create_engine('sqlite:///letters.db')
connection = engine.connect()

inspector = sa.inspect(engine)
inspector.get_table_names()

import pandas as pd
df_img = pd.read_sql('SELECT * FROM images', con=connection)
df_labels = pd.read_sql('SELECT * FROM labels', con=connection)

df_img.index = df_img.loc[:, 'index']
df_img = df_img.drop('index', axis=1)

df_labels.index = df_labels.loc[:, 'index']
df_labels = df_labels.drop('index', axis=1)

print(df_img.columns)
print(df_labels.columns)


# make all imports
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# define figure with 10 axes in 2 rows
fig, axs = plt.subplots(nrows=2, ncols=5, figsize=[20,4])  # define figure with 10 axes in 2 rows
# plot images of C in the first row
for i in range(5):
    axs[0, i].imshow(df_img.loc[i, :].values.reshape(28,28), cmap='gray')  # use cmap to get nice black and white pictures
# plot images of I in the second row
for i in range(1, 6):  # we use .iloc and negative values. Negative 0 is also 0, so have to start with 1
    axs[1, i-1].imshow(df_img.iloc[-i, :].values.reshape(28,28), cmap='gray')  # I letters are at the bottom of the DataFrame



# Display high-dimensional data in two dimensions
from sklearn.decomposition import PCA
pca = PCA(n_components=2, random_state=10)
arr_img = df_img
arr_img_2d = pca.fit_transform(arr_img)

sum(pca.explained_variance_ratio_)

import seaborn as sns
sns.scatterplot(x=arr_img_2d[:, 0], y=arr_img_2d[:, 1], hue=df_labels.loc[:, 'labels'], alpha=0.3)

fig, axs = plt.subplots(ncols=2, figsize=[4,8])  # define figure with 2 axes
axs[0].imshow(pca.components_[0].reshape(28, 28), cmap='gray')  # visualize the first principal component (PC1)
axs[1].imshow(pca.components_[1].reshape(28, 28), cmap='gray')  # visualize the second principal component (PC2)
axs[0].set_title('PC1')  # set titles for the axes
axs[1].set_title('PC2')


# Defining the number of principal components using the content of the information

pca = PCA(random_state=10)
pca.fit(arr_img)

import numpy as np
cumulative_sum_ratios = np.cumsum(pca.explained_variance_ratio_)
cumulative_sum_ratios[:3]

# we determined the number of clusters in the last chapter with the elbow method
# We propose the percentage of variance based on the number of components. pca.explained_variance_ratio_
# This is always the proportion of the total variance that can be explained by the respective component. 
# However, the best way to do this is to use the cumulative sum of these values. For two features we can explain the proportion given by the sum of the first two values.
# We can calculate this cumulative sum 

import numpy as np
cumulative_sum_ratios = np.cumsum(pca.explained_variance_ratio_)
cumulative_sum_ratios[0:5]

# The result is the Explanied variance ratio per each component 
# array([0.33416344, 0.55857857, 0.6142112 ])

fig, ax = plt.subplots(1, figsize=(8, 8))
ax.plot(range(1, len(cumulative_sum_ratios)+1), # we have at least 1 component
        cumulative_sum_ratios)
ax.set(title='Explained Variance with PCA',
       xlabel='Number of PCs',
       ylabel='Cumulative sum of explained Variance ratio');

# We can specify values between 0 and 1 for n_components
# then chooses the number of components so that at least this portion of the variance can be reproduced by the components. 

pca = PCA(n_components=.98)
pca.fit(arr_img)
len(pca.components_)

# fit pca
pca = PCA(n_components=0.98)
arr_img_pca = pca.fit_transform(arr_img)

# reconstruct the data
arr_img_reconstructed = pca.inverse_transform(arr_img_pca)  # reverse pca

# define figure with 10 axes in 2 rows
fig, axs = plt.subplots(nrows=2, ncols=5, figsize=[20, 4])  # define figure with 10 axes in 2 rows
# plot images of original C in the first row
for i in range(5):
    axs[0, i].imshow(df_img.loc[i, :].values.reshape(28, 28), cmap='gray')  # use cmap to get nice black and white pictures
# plot images of reconstructed C in the second row
for i in range(5):
    axs[1, i].imshow(arr_img_reconstructed[i, :].reshape(28, 28), cmap='gray')  # reconstructed values are in arr_img_reconstructed 


# Remember:
# Dimensionality reduction helps to visualize high-dimensional data and can make its structure more understandable
# Pass a number between 0 and 1 to n_components to determine the number of components divided by the proportion of the declared variance
# A visualization of the explained variance can help you to choose n_components


