# Silhouette scores

# import the data from pickle
import pandas as pd
df_customers = pd.read_pickle('customer_data_prepared.p')

# import, instantiate and fit StandardScaler
from sklearn.preprocessing import StandardScaler
standardizer = StandardScaler()
standardizer.fit(df_customers)

# standardize the data
arr_customers_std = standardizer.transform(df_customers)

df_customers.head()


from sklearn.cluster import KMeans

model = KMeans(n_clusters=7, random_state=0)
model.fit(arr_customers_std)




# Where mean\_dist_cc is the average distance to the points of the closest cluster. 
#    mean\_dist_mc is the average distance to the points in the same cluster (*my cluster*). 
# The maximum function in the divider only selects the maximum of the two average distances

# If  silhouette_coef
#   is positive, the point is closer to its own* The higher the value, the more typical the point is for the cluster.
# If  silhouette_coef
#   is negative, the point is further away from its own cluster. It could have been assigned to the wrong cluster.
# A score of close to zero implies that the data point is located equidistant between the clusters. This may be because the clusters overlap.



from sklearn.metrics import silhouette_score
silhouette_score(   arr_customers_std,
                    model.labels_,
                    metric='euclidean')


from sklearn.metrics import silhouette_samples
import numpy as np

arr_sil = silhouette_samples(arr_customers_std,
                            model.labels_,
                            metric='euclidean')
print(np.min(arr_sil))

mask = np.min(arr_sil) == arr_sil

print('Minimal coefficent:', arr_sil[mask])
print('In cluster:', model.labels_[mask])
df_customers.iloc[mask, :]

# We can try this out by calculating the average distance from this point to the data points in the other clusters. 
from sklearn.metrics import euclidean_distances

mask = np.min(arr_sil) == arr_sil  # create a mask to select the data point with the worst silhouette score
for cluster in range(7):  # iterate through every cluster
    arr_cluster = arr_customers_std[model.labels_ == cluster]  # select the values of this cluster
    distances = euclidean_distances(arr_cluster, arr_customers_std[mask].reshape(1, -1))  # calculate the distances to the point with the worst silhouette score
    print(cluster,':', np.mean(distances))  # calculate and print the mean


for cluster in range(7):  # iterate through every cluster
    distances = euclidean_distances(model.cluster_centers_[cluster].reshape(1, -1), arr_customers_std[mask].reshape(1, -1))  # calculate the distances between the point with the worst silhouette score and the cluster center
    print(cluster,':', np.mean(distances))



# Visualizing the silhouette scores
import matplotlib.pyplot as plt
mask = model.labels_ == 0
sv_len = len(arr_sil[mask])

sv_sorted = np.sort(arr_sil[mask])

fig, ax = plt.subplots()
ax.barh(range(sv_len), sv_sorted)
plt.title("Silhouette scores of cluster {}".format(str(0)))
plt.ylabel("Rank in Cluster")
plt.xlabel("Silhouette score");



fig, ax = plt.subplots(figsize=(15, 15))
start = 0
end = 0

for cluster in range(7):
    mask = model.labels_ == cluster

    sv_len = len(arr_sil[mask])
    sv_sorted = np.sort(arr_sil[mask])

    end = end + sv_len
    ax.barh(range(start, end), width=sv_sorted, label = cluster)
    start = end
ax.legend()
ax.set_title("Silhouette scores for k-Means with k = {}".format(cluster))

fig, axs = plt.subplots(nrows = 5,
                        figsize=(15, 50),
                        sharex=True)  # create figure with 5 axes because we visualize 5 clusters

for n_cluster in range(3,8):
    # fit new KMeans with n_clusters
    model = KMeans(n_clusters=n_cluster, random_state=0)
    model.fit(arr_customers_std)
    #calculate silhouette coefficient and scores
    print('Number of clusters:',n_cluster,'overall score:', silhouette_score(arr_customers_std, model.labels_))
    arr_sil = silhouette_samples(arr_customers_std, model.labels_)
    
    #plot the scores
    start = 0  # start and end are needed to plot one group above the other
    end = 0
    for cluster in range(n_cluster):
        mask = model.labels_ == cluster  # create a mask to select data points from the current cluster
        sv_len = len(arr_sil[mask])   # the length is needed to increase end for plotting the next cluster above this one
        sv_sorted = np.sort(arr_sil[mask])  # sort the silhouette scores within each cluster to have a nicer plot
        end = end + sv_len  # increase end: be able to get a range with the length of this cluster
        axs[n_cluster-3].barh(range(start, end), sv_sorted, label=cluster)  # plot the silhouette scores of this cluster
        start = end  # increase start: the next cluster will be plotted above this one
    axs[n_cluster-3].set_title("Silhouette scores for k-Means with k = {}".format(n_cluster))
    axs[n_cluster-3].legend()
    plt.xlim([-1,1])



# Now you know two numerical values to describe the quality of our clusters: The average within-cluster sum of squares and 
# the average silhouette score, known as the silhouette coefficient.

# With the within-cluster sum of squares, only the distance of each data point to its own cluster center is taken into account. 
# The distance to other clusters is ignored. It is not standardized. So if the clusters contain more data points, this value also increases. 
# Therefore the value of the within-cluster sum of squares is not suitable for comparing different data sets. We should only use it to 
# determine the number of clusters with the elbow method.

# With the silhouette method, the distance to the cluster center is not directly included in the calculation. 
# Instead, this metric focuses on the distance between the data points of one cluster and the nearest cluster.
# If the distances between the clusters increase, the average silhouette coefficient increases. 
# It is also standardized to a value between -1 and 1. This value can be used to compare the structure of different data sets. 
# The higher the average silhouette coefficient, the more isolated the clusters making up the data point are. 
# Furthermore, you can use the silhouette scores for the individual data points to check whether a specific point has been well assigned.

# Remember
# The silhouette coefficient characterizes data points using the distance between points in one cluster and the nearest cluster
# Negative scores show that a data point has possibly been incorrectly assigned
# If there are large, clear differences, these have a high positive score
# You can use the silhouette method to estimate the number of clusters