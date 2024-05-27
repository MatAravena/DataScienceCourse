import pandas as pd
df_composition = pd.read_csv('wine_composition.csv')
df_composition
df_composition.info()
df_composition.loc[:, 'color'] = df_composition.loc[:, 'color'].astype('category')
df_composition.isna().sum()

df_rating  = pd.read_csv('wine_rating.csv')
df_rating.head()

df_rating.isna().sum()

sorted(df_rating.loc[:,'rating'].unique())

from sklearn.linear_model import LinearRegression
model = LinearRegression()

sum(df_composition.loc[:, 'id'] != df_rating.loc[:, 'id'])

target = df_rating.loc[:, 'rating']
features = df_composition.drop(['color','id'], axis=1)



# cross_val_score(estimator=model,         # the model that will make the predictions
#                 X=features            # the feature matrix
#                 y=target               # the target vector
#                 cv=int                  # the number of folds to be used for cross-validation
#                 scoring=str or function # the metric that rates the result
#                )
from sklearn.model_selection import cross_val_score
cv_results = cross_val_score(estimator=model, X=features, y=target, cv=5, scoring='neg_mean_absolute_error')
cv_results.mean()


# Creating features of a higher degree

# If you have no idea which features can help, it's often a good idea to look at the interactions and higher degrees of features.
# This means the paired products of the features and for example the squared values of the features in pairs.
# The linear regression generates the predictions just by weighting and summing up the features. It is therefore not able to create products from the features.

# PolynomialFeatures(degree=int,           # The degree of the resulting polynomial.
#                    interaction_only=bool # Controls whether self interactions are included.
#                    include_bias=bool     # Controls whether the 1 is also included as a feature.
#                   )


# The most important hyperparameter of PolynomialFeatures is degree. 
# This specifies the degree the new features should have. include_bias specifies whether a column consisting only of ones should be included. 
# Normally we don't need this, because LinearRegression already takes care of this internally (with the fit_intercept parameter). 
# The parameter interaction_only controls whether only the interactions should be generated. For example,  (𝑓𝑒𝑎𝑡𝑢𝑟𝑒_1)⋅(𝑓𝑒𝑎𝑡𝑢𝑟𝑒_2)
#   would be an interaction, while  (𝑓𝑒𝑎𝑡𝑢𝑟𝑒_1)2   would not.


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

# Our predictions have improved by about `0.004` on average. This may not sound like too much, but note that the amount of work involved was 
# very small, so it is often preferable to use polynomial regression rather than linear regression.

# Remember:

# Feature engineering describes the process of creating new features that can help when making predictions
# Create features of a higher degree with the PolynomialFeatures transformer.
# A linear regression with features of a higher degree is a polynomial regression.
