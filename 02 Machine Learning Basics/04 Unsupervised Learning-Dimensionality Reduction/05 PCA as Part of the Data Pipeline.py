import pandas as pd
df_composition = pd.read_csv('wine_composition.csv')  # import the composition of the wine data
df_rating = pd.read_csv('wine_rating.csv')  # import the ratings for the wines

df_composition.head()



# import and create the regression model
from sklearn.linear_model import LinearRegression
model = LinearRegression()

target = df_rating.loc[:, 'rating']  # do not use id for target variable
features = df_composition.iloc[:, :-2]  # only use numeric values as features

# use cross-validation to evalute the model
from sklearn.model_selection import cross_val_score
cv_results = cross_val_score(estimator=model, X=features, y=target, cv=5, scoring='neg_mean_absolute_error')  # do 5-fold cross validation to evaluate the linear regression
print('negative mean absolute error:', cv_results.mean())


# negative mean absolute error: -0.58104275094183



from sklearn.preprocessing import PolynomialFeatures
poly_transformer = PolynomialFeatures(degree=2, include_bias=False)

from sklearn.pipeline import Pipeline
pipeline = Pipeline([('poly', poly_transformer), 
                     ('reg', model)])

cv_results = cross_val_score(estimator=pipeline,
                             X=features,
                             y=target,
                             cv=5,
                             scoring='neg_mean_absolute_error'
                            )
cv_results.mean()


# negative mean absolute error: -0.5771691803265556


# With Poly poly_transformer
featuresTransformed =  poly_transformer.fit_transform(features)
featuresTransformed.shape

# Relate the transformer to the pipeline

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

pipeline = Pipeline([('poly', poly_transformer),
                     ('scale', StandardScaler()),
                     ('pca', PCA()),
                     ('reg', model)])


# validation_curve(estimator=model, # the model to be used (`pipe`)
#                  X=DataFrame,     # the feature matrix (`features`)
#                  y=DataFrame,     # the target vector (`target`) 
#                  param_name=str,  # the hyperparameter to be varied within a part of the pipeline (`'pca__n_components'`)
#                  param_range=list,# the numbers to be used for the parameter (`range(1,50)`)
#                  cv=int           # the number of cross validation folds (`5`)
#                  scoring=str      # the evaluation metric (`'neg_mean_absolute_error'`)

from sklearn.model_selection import validation_curve

train_scores, valid_scores = validation_curve(estimator=pipeline,  # estimator (pipeline)
                                              X=features,  # features matrix
                                              y=target,  # target vector
                                              param_name='pca__n_components',  # define model hyperparameter (k)
                                              param_range=range(1,50),  # test these k-values
                                              cv=5,  # 5-fold cross-validation
                                              scoring='neg_mean_absolute_error')  # use negative validation

import numpy as np
train_scores_mean = np.mean(train_scores, axis=1)
valid_scores_mean = np.mean(valid_scores, axis=1)

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8,8))
ax.plot(range(1,50), train_scores_mean, label='train')
ax.plot(range(1,50), valid_scores_mean, label='valid')
ax.hlines(y=-0.57717, xmin=1, xmax=50)
ax.set_xlabel('Number of components')
ax.set_ylabel('Negative mean absolute error')
ax.legend();



print('best score:', max(valid_scores_mean))
print('position:', valid_scores_mean.tolist().index(max(valid_scores_mean)))

# best score: -0.5732229845114517
# position: 25




# Interpreting the principal components
pipeline = Pipeline([('poly', poly_transformer), 
                     ('scale', StandardScaler()), 
                     ('pca', PCA(n_components=26)), 
                     ('reg', model)])

pipeline.fit(features,target)

pca = pipeline.named_steps['pca']


# To see with PCA is more important with the higher Explained Variance
import seaborn as sns
#create names
pc_names = ['pc{}'.format(i+1) for i in range(pca.n_components)]

#plot scree plot
ax = sns.barplot(x=pc_names, y=pca.explained_variance_ratio_)

#style plot
ax.set(title="Scree plot", ylabel="Explained Variance")
plt.xticks(rotation=90);

# Let's take a look at these recipes for pca. You can find them in the attribute my_pca.components_, which are called eigenvectors
eigenvectors = pd.DataFrame(pca.components_, index=pc_names)

# To check the names of the features from the initial DataFrame
PolynomialFeatures = pipeline.named_steps['poly']
poly_feature_names = poly.get_feature_names(input_features=features.columns)
eigenvectors.columns = poly_feature_names

# Now is check the names and the relation with are more importante
plt.figure(figsize=(15,5))
ax = eigenvectors.loc['pc1',:].sort_values(ascending=False).plot.bar()
ax.set(title="Eigenvector for pc1", ylabel="Importance in pc1");

# And apparently the first 10 are more important
plt.figure(figsize=(15,5))
ax = abs(eigenvectors).loc['pc1',:].sort_values(ascending=False).head(10).plot.bar()
ax.set(title="Eigenvector for pc1", 
       ylabel="Importance in pc1");


# Remember:

# Validation curves can help you get a feel for the number of principal components to improve predictions
# First create the polynomial features, then standardize them and then you can use 'PCA'
# Use my_pca.components_ to evaluate the principal components. Here you can find the corresponding parts of the features that make up the principal components.
