import pandas as pd
df_customers = pd.read_pickle('customer_data_prepared.p')

df_customers.info()
df_customers.head()


#suppress conversion warning
from sklearn.exceptions import DataConversionWarning
import warnings
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

print(df_customers.columns)

#df_customers =pd.merge(df_customers.loc[:,['Revenue', 'Quantity']], 
#                       df_customers.loc[:, ['InvoiceNo','RevenueMean', 'QuantityMean',
#                           'PriceMean', 'DaysBetweenInvoices', 'DaysSinceInvoice']], 
#                       left_index=True, 
#                       right_index=True, 
#                       how='left')

from sklearn.preprocessing import StandardScaler
standardizer = StandardScaler()
standardizer.fit(df_customers)
arr_customers_std = standardizer.transform(df_customers)
arr_customers_std[0, :]

from sklearn.cluster import KMeans
model = KMeans(n_clusters=3)
model.fit(arr_customers_std)
model.score(arr_customers_std)

# Important
# The higher this value, the further away the data points are from the cluster center. So a smaller value is better


# k-Means offers us what's called the within-cluster sum of squares (WCSS)
from numpy.linalg import norm  # import norm to calculate the distance

wcss = 0  # define the distance
for cluster in range(model.n_clusters):  # itereate through every cluster
    cluster_center = model.cluster_centers_[cluster]  # select the cluster center
    cluster_mask = model.labels_ == cluster  # create a mask to get only customers belonging to the selected cluster
    for row in arr_customers_std[cluster_mask, :]:  # iterate through every data point in this cluster
        distance = norm(cluster_center - row)**2  # calculate the distance from row to customer using norm and square the distance
        wcss = wcss + distance  # add the squared distance to the within-cluster sum of squares
print(wcss)



# Finding the right amount of clusters
cluster_scores = []
for x in range(1, 30):
    modelTry = KMeans(n_clusters=x)
    modelTry.fit(arr_customers_std)
    cluster_scores.append(modelTry.score(arr_customers_std))
    print(x, '-', modelTry.score(arr_customers_std))

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

fig, ax = plt.subplots()
ax.plot(range(1, 30), cluster_scores, marker='o', markersize=7)
ax.set_title("Finding the best value for k")
ax.set_xlabel("k")
ax.set_ylabel("Within-cluster sum of squares")

plt.tight_layout()



# Interpreting the clusters
import numpy as np

for x in [1,10,50]:
    anotherClusterTest= []

    for y in range(10):
        model = KMeans(n_clusters=7, n_init=x)
        model.fit(arr_customers_std)
        anotherClusterTest.append(model.score(arr_customers_std))
    print(x)
    print( np.mean(anotherClusterTest) , np.std(anotherClusterTest), '\n')



# The clusters are labelled at random. So if you cluster the same data twice and you get basically the same clusters, they might have different names
# For example, if we used random_state=42 instead of random_state=0, the indexing of the clusters might change. This can cause confusion. 
# So you should always specify the random_state when you need to be able to reproduce the results
model = KMeans(n_clusters=7, random_state=0)
model.fit(arr_customers_std)
model.score(arr_customers_std)

df_customers.loc[:, 'Labels'] = model.labels_
pd.crosstab(df_customers.loc[:, 'Labels'], columns='count')

mask = df_customers.loc[:, 'Labels'] == 3
df_customers.loc[mask, :]

groups_customers = df_customers.groupby('Labels')

# The median is the value that is exactly in the middle when all values of a feature are ordered by size.
# This means that one half of the values is less than or equal to the median and the other half is greater than or equal to the median. 
# The median is therefore often a good choice for quickly assessing the results.
groups_customers.median()

# IMPORTANT
# Check the PDF to understand better how to interpretate the data

# Remember:
# The KMeans score describes the quality of the clustering with the average sum of the squared Euclidean distances from the data points to the cluster centers
# Make sure features are on the same scale before clustering
# Determine the number of clusters with the elbow method
# Make results reproducible by using random_state
