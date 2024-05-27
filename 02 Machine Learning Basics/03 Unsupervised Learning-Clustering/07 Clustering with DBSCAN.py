# import everything we need
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
import matplotlib.pyplot as plt
import seaborn as sns

# read data
spherical_cluster = pd.read_csv('clustering-data.csv')

# define figure and colors
fig, ax = plt.subplots(figsize=[15,10])
colors = sns.dark_palette("#3399db",4)  # seaborn can be used to derive a color palette from one color

# fit new KMeans with n_clusters
model = KMeans(n_clusters=4, random_state=0)
model.fit(spherical_cluster)
#calculate silhouette score and coefficients
arr_sil = silhouette_samples(spherical_cluster, model.labels_)

#plot the coefficients
start = 0
end = 0
for c in range(4):
    mask = model.labels_ == c
    sv_len = len(arr_sil[mask])
    sv_sorted = np.sort(arr_sil[mask])
    end = end + sv_len
    ax.barh([i for i in range(start, end)], width=sv_sorted, label=c, color=colors[c])
    start = end
ax.set_title('Silhouette scores of k-Means clustering with spherical data', size=16, loc='left')
ax.set_xlabel(xlabel='silhouette score',
              position=[0, 0],
              horizontalalignment='left',
              size=14)
ax.set_ylabel(ylabel='data point',
              position=[0, 1],
              horizontalalignment='right',
              size=14)
ax.legend()

other_cluster = pd.read_csv('clustering-data-2.csv')

model = KMeans(n_clusters=2, random_state = 0)
model.fit(other_cluster)

# plot the data:
import seaborn as sns
sns.scatterplot(data = other_cluster, x='x', y='y', hue=model.labels_).set_title("k-Means Clustering with k=2");


# These are all related to the fact that k-Means only assesses the distances between data points and cluster centers:
# 1. k-Means has problems with values on different scales
# 2. k-Means can't determine the number of clusters itself
# 3. k-Means can't deal with clusters whose shapes differ widely from that of a circle or sphere

# In the last few lessons, you learned how to deal with the first two weaknesses by standardizing the data and using various scores for the number of clusters. 


# DBSCAN works as follows:

# A maximum distance eps is defined.
# All points that have at least min_samples of neighbors within this radius, are what are known as core points of a cluster
# Data points that are close enough to core points but don't have enough neighbors themselves are called border points
# Neighboring core and directly reachable points are assigned to the same cluster
# Data points that lie outside a cluster's neighborhood that don't have enough neighbors themselves to form their own cluster, are known as outliers or noise points

# The circles have a radius of eps=0.12.
# min_samples=4. This means that 4 points are sufficient to form a cluster







# But how do you now decide which values are suitable for eps and min_samples? 
# As with the number of clusters in KMeans you have to estimate it. 
# When choosing a minimum distance, it can be helpful to visualize the distances in the data set.

# The euclidean_distances returns a two-dimensional array with the distances between all data points.

from sklearn.cluster import DBSCAN
model_db = DBSCAN(eps=0.12, min_samples=4)
model_db.fit(other_cluster)
model_db.labels_

sns.scatterplot(data = other_cluster, x='x', y='y', hue=model_db.labels_).set_title("k-Means Clustering with k=2");


from sklearn.metrics import euclidean_distances
arr_dist = euclidean_distances(other_cluster)

arr_dist_sorted = np.sort(arr_dist, axis=1)

sns.displot(arr_dist_sorted[:,1],kde = False )


# So min_samples is about twice the number of columns in the data. So in our example 
# it's 4, because we only have the columns 'x' and 'y' in the data. But the exact choice depends on the 
# result. A small value can result in the data being split into a large number of clusters. If the value is large, you can end up with a lot of outliers.

arr_sil = silhouette_samples(other_cluster, model_db.labels_)

# define figure and colors
fig, ax = plt.subplots(figsize=[15, 10])
colors = sns.dark_palette("#3399db", 2)  # seaborn can be used to derive a color palette from one color, we use this to color the clusters

# plot the coefficients
start = 0  # start and end are needed to plot one group above the other
end = 0
for cluster in range(2):
    mask = model_db.labels_ == cluster  # create a mask to select data points from the current cluster
    sv_len = len(arr_sil[mask])  # the length is needed to increase end for plotting the next cluster above this one
    sv_sorted = np.sort(arr_sil[mask])  # sort the silhouette scores within each cluster to have a nicer plot
    end = end + sv_len  # increase end: be able to get a range with the length of this cluster
    ax.barh(range(start, end), width=sv_sorted, label=cluster, color=colors[cluster])  # plot the silhouette scores of this cluster
    start = end  # increase start: the next cluster will be plotted above this one
    
# set title and labels of the plot
ax.set_title('Silhouette scores of DBSCAN with crescent-shaped clusters', size=16, loc='left')
ax.set_xlabel(xlabel='silhouette score',
              position=[0, 0],
              horizontalalignment='left',
              size=14)
ax.set_ylabel(ylabel='data point',
              position=[0, 1],
              horizontalalignment='right',
              size=14)
ax.legend()

# A lot of the data points have a negative score. But we know that they were correctly classified. The *silhouette* method is not suitable for this data set, 
# because the clusters are not circular and the distance between them is very small in some places.


# You've performed your first cluster analysis with DBSCAN. Now you have added a second clustering algorithm to your repertoire. 
# DBSCAN is a very good addition to k-Means, because they both proceed very differently.


from sklearn.metrics import euclidean_distances
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# read the data
import pandas as pd
df_customers = pd.read_pickle('customer_data_prepared.p')

# standardize the values
from sklearn.preprocessing import StandardScaler
standardizer = StandardScaler()
standardizer.fit(df_customers)
arr_customers_std = standardizer.transform(df_customers)

arr_dist = euclidean_distances( arr_customers_std )
arr_dist_sorted = np.sort(arr_dist, axis=1)

fig, ax = plt.subplots()
ax.plot(range(len(arr_dist_sorted)), np.sort(arr_dist_sorted[:, 16]))
ax.set(title='Sorted distances to 16th neighbor', xlabel='Data points', ylabel='Distance to 16th neighbor')

# You can often use what's called a k-distance plot to help estimate the distance. 
# Here the distance to the kth nearest neighbor is displayed for each data point. 
# The idea behind this is that data points in a cluster have many neighbors. 
# The distance therefore only changes slightly from the nearest neighbor to the nearest but one. 
# But if we look at a point outside the cluster, this is no longer the case. The neighbors are far away and the distance increases from one to the next.

# In the diagram, we would therefore expect that all points in clusters to have a small distance to their kth neighbor. 
# However, this distance rises sharply for data points outside the cluster. So for eps you choose a value which lies just 
# before the sudden increase. k corresponds to min_samples. Usually the images are relatively stable for small deviations in k.

fig, ax = plt.subplots()
ax.plot(range(len(arr_dist_sorted)), np.sort(arr_dist_sorted[:, 16]))
ax.set(title='Sorted distances to 16th neighbor', xlabel='Data points', ylabel='Distance to 16th neighbor')

np.sort(arr_dist_sorted[:, 16])[2798]

model = DBSCAN(eps=2.056, min_samples=16)
model.fit(arr_customers_std)
np.unique(model.labels_)

sum(model.labels_ == -1)

from sklearn.decomposition import PCA
import seaborn as sns

#pca
pca = PCA(n_components=2)
cluster_plot_arr = pca.fit_transform(arr_customers_std)

#plot
ax = sns.scatterplot(x=cluster_plot_arr[:,0],
                     y=cluster_plot_arr[:,1],
                     alpha=0.7,
                     hue=model.labels_)
#style plot
ax.set(xlabel='PC1',
       ylabel='PC2',
       title='Cluster assignment (dimensionality reduced)')

#configure legend
import matplotlib.pyplot as plt


plt.legend(title='Clusters',
           labels=['Outliers', 'Cluster 0']);


# Congratulations: Now you know the KMeans and DBSCAN algorithms. You know how to set values for their parameters. 
# You have also seen how the data must be made up for these algorithms to work well.

# For the customer analysis you applied 2 clustering algorithms which follow different methods, 
# but produce similar results with this data set. 
# Your employer thanks you for your assessment of the cluster analysis. 
# He decides that the next analysis will be a much larger project, which should include even more of the customers' features.

# Remember:

# DBSCAN forms clusters based on the density of data points
# 2⋅𝑛𝑢𝑚𝑏𝑒𝑟_𝑜𝑓_𝑑𝑖𝑚𝑒𝑛𝑠𝑖𝑜𝑛𝑠
#   is a good guiding value for min_samples
# Estimate eps with a k-distance plot, which shows the distances to the nearest neighbors.
# KMeans and the silhouette method work best with spherical clusters