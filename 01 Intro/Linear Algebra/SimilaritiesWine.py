import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
df = pd.read_csv('wine-quality.csv')
df.head()


sns.set(font_scale=1.5) #make fonts bigger
sns.pairplot(data=df, hue='color', palette=['#70b8e5','#e54f59']);

sns.scatterplot(data=df, x='residual.sugar', y='sulphates', hue='color',  palette=['#70b8e5','#e54f59']);

import numpy as np
from numpy.linalg import norm

cols = ['residual.sugar', 'sulphates']
# Take first position of the vectors 
vec1 =  df.loc[0,cols]
vec2 =  df.loc[1,cols]

# Calculate the Scalar product or the Dot roduct
# np.dot(vec1 ,vec2) 
# Then take the length relation for the vectors
# norm(vec1) 

np.dot(vec1 ,vec2) / (norm(vec1) *  norm(vec2)) 
# We get about 0.96 for similarity.

cols = ['residual.sugar','sulphates']  # only use these columns
similarity = np.dot(df.loc[0,cols],df.loc[1,cols])  # calculate the dot product
similarity = similarity / (norm(df.loc[0, cols]) * norm(df.loc[1,cols]))  # divide by the product of the vector lengths



def wine_recommendation(df_wines, wine_id, sim_min):
    """Return positions of wines in df_wines, which are similar to the wine on position wine_id.

    Use cosine-similarity as measure of likeness between two wines.
    
    Args:
        df_wines (DataFrame): Contains the wines-quality data.
        wine_id (int): Position inside df_wines of the wine, on which the recommendation is based.
        sim_min (float): Controlls the minimum similarity wines must have to be recommended. Only values between -1 and 1 should be used.

    Returns:
        recommendations (list): Contains the positions of wines that are similar to wine_id with a value of at least sim_min.
        
    """

    recommdations = []
    vec =  df_wines.loc[wine_id,cols]

    for i in range(len(df_wines)):
        vec2 =  df_wines.loc[i,cols]
        similarity = np.dot(vec2 ,vec) / (norm(vec2) *  norm(vec) )

        if similarity >= sim_min:
            recommdations.append(i)
    return recommdations

recommendation_0 = wine_recommendation(df,0,0.99999)
recommendation_6400 = wine_recommendation(df,6400,0.99999)
print(recommendation_0)
print(recommendation_6400)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[6, 6])
sns.scatterplot(data=df.loc[recommendation_0, :],    hue='color', x='residual.sugar', y='sulphates', palette=['#70b8e5'], ax=ax)  # palette defines the color blue
ax.set_title('Wines similar to wine_id = 0')

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[6, 6])
sns.scatterplot(data=df.loc[recommendation_6400, :], hue='color', x='residual.sugar', y='sulphates', palette=['#70b8e5', '#e54f59'], ax=ax)  # palette defines colors blue and red
ax.set_title('Wines similar to wine_id = 6400')

# Remember:

# Quantify the similarity of two products by the angle of their vectors with cosine similarity.
# Define your own functions with def
# Add a docstring to your own functions so that you and other people using your function understand what your function does