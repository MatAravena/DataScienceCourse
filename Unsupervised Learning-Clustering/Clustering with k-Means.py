import pandas as pd
df = pd.read_csv('clustering-data.csv')
df.head()


import seaborn as sns
sns.scatterplot(x=df.loc[:,'x'],  y=df.loc[:,'y'])



# The most important hyperparameter for k-means is n_clusters. This is actually k that gives the algorithm its name, and it specifies the number of clusters
from sklearn.cluster import KMeans
model = KMeans(n_clusters=4)

model.fit(df)
#print('cluster_centers_', model.cluster_centers_)
#print('labels_', model.labels_)

sns.scatterplot(x=df.loc[:,'x'],  y=df.loc[:,'y'], hue=model.labels_)

print('shape', df.shape)


# Since the cluster centers are always moved to the centers of their associated points (step 4 of k-means), 
# we would expect them now to be in the middle of the visible clusters. Their 'x' and 'y' values should be roughly the same as the averages of each group. 
df.loc[:,'labels'] = model.labels_
print(df.groupby('labels').mean())
print(model.cluster_centers_)


ax = sns.scatterplot(x=df.loc[:,'x'],  y=df.loc[:,'y'], hue=model.labels_, markers=200)
sns.scatterplot(markers=200, ax=ax, s=200, 
                x=pd.DataFrame(model.cluster_centers_).iloc[:,0],
                y=pd.DataFrame(model.cluster_centers_).iloc[:,1])


# Compare predicted with the model labels created previosly
labels_predicted = model.predict(df.loc[:, ['x', 'y']])
sum(labels_predicted != model.labels_)



# We have made sure that KMeans really does use the L2 norm to calculate the distance. 
# The values with the smallest Euclidean distance to a cluster center are assigned to this center
from sklearn.metrics.pairwise import euclidean_distances
modelTransformed =  model.transform(df.loc[:, ['x','y']])
euclidian = euclidean_distances(df.loc[:,['x','y']], model.cluster_centers_)
sum(modelTransformed != euclidian)

# Remember:

# k-means assigns every datapoint to the closest cluster center, using the L2 norm to calculate the distance
# n_clusters returns the number of cluster centers
# Cluster centers are moved to the mean values of the corresponding data points
# Create a scatter plot with different colored groups with sns.scatterplot() and its hue parameter
