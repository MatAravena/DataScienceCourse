import numpy as np
from numpy.linalg import norm


# Calclate the length of a vector
vec = np.array([13,52])
vec2 = np.array([113,52])
vec_length = norm(vec2 - vec)
print(vec_length)


# The norm() function calculates the length of a vector, which we can use to measure a distance! If you don't 
# specify additional arguments it calculates the Euclidean norm. This is the norm we use in everyday life to measure distances. 
# It is also called the L2 norm. Remember this term, because it comes up a lot in the context of machine learning.

# Our distinction between metric and norm is somewhat crude. There are also precise mathematical definitions for both of them. But don't worry, you don't 
# need to learn them now. The most important thing is that you connect the terms norm and metric with measuring lengths and distances of vectors.



import pandas as pd
df = pd.read_csv('wine-quality.csv')
df.head()

distance_idx0 = []
for i in df.index:
    distance_idx0.append( norm(df.iloc[i,:-1] - df.iloc[0,: -1]))

print(distance_idx0[:2]) 



import matplotlib.pyplot as plt
fig, ax = plt.subplots(nrows=1,ncols=1)
ax.hist(distance_idx0, bins=50)
ax.set(xlabel='Similarity based on L2 metric',ylabel='Number of wines')



def wine_recommendation_L2(df_wines, wine_id, quantity):
    """Return positions of wines in df_wines, which are similar to the wine at position wine_id.
    
    Use L2 norm as a measure of similarity between two wines.
    
    Args:
        df_wines (DataFrame): Contains the wines quality data.
        wine_id (int): Position in df_wines of the wine, on which the recommendation is based.
        quantity (int): Number of recommended wines to return.

    Returns:
        recommendations (list): Contains the positions of quantity wines that are most similar to wine_id.
        
    """
    cols = df_wines.columns[:-1] 
    distances = []
    for i in range(len(df_wines)):
        distances.append( norm( df_wines.loc[i,cols] - df_wines.loc[wine_id,cols] ))
    return np.argsort(distances)[:quantity]



wine_recommendation_L2(wine_id = 0, quantity = 1, df_wines = df)
wine_recommendation_L2(wine_id = 3200, quantity = 1, df_wines = df)
wine_recommendation_L2(wine_id = 6400, quantity = 1, df_wines = df)



recommendation_0 = wine_recommendation_L2(wine_id = 0, quantity = 10, df_wines = df)
df.loc[recommendation_0, :]



# Additional metrics
#Doing the real math formula for L2 norm
# so en otras palabras es calcular la relacion de los 2 puntos dentro de un vector

import numpy as np
def lp_norm( vec, p):
    correlation = 0
    for i in range(0, len(vec)):
        correlation += abs(vec[i])**p
    return correlation**(1/p)

print('L1 norm for vec: ', lp_norm(vec=[2,1], p=2))
print('L2 norm for vec: ', lp_norm(vec=[2,1], p=1))

def lp_norm2(vec, p):
    vec_p = [abs(x)**p for x in vec]
    return (np.sum(vec_p))**(1/p)

print('L1 norm for vec: ', lp_norm2(vec=[2,1], p=2))
print('L2 norm for vec: ', lp_norm2(vec=[2,1], p=1))






import matplotlib.pyplot as plt

vec = [2, 1]
x = range(1,10)
y = [lp_norm(vec, i) for i in x]

plt.plot(x,y)



# Remember:

# Calculate the distance between two vectors with the L2 norm with norm(my_array_1 - my_array_2)
# Metrics and norms are associated with distances
# These can be used as a measure for similarly
# Use docstrings and function tests for your own functions
