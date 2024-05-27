import pandas as pd
df = pd.read_excel('Taiwan_real_estate_training_data.xlsx', index_col='No')
col_names = ['house_age', 
              'metro_distance', 
              'number_convenience_stores', 
              'number_parking_spaces',
              'air_pollution',
              'light_pollution',
              'noise_pollution',
              'neighborhood_quality',
              'crime_score',
              'energy_consumption',
              'longitude', 
              'price_per_ping']
df.columns = col_names
df.loc[:, 'price_per_m2'] = df.loc[:, 'price_per_ping'] / 3.3


# multiple linear regression in five steps:

# Select model type: Linear Regression
# Instantiate the model with certain hyperparameters: Store standard model with axis intercept in model_multiple
# Organize data into a feature matrix and target vector: features and target
# Model fitting: use my_model.fit()
# Make predictions with the trained model: use my_model.predict()
# We'll stick with LinearRegression from the sklearn.linear_model module. Import it.


from sklearn.linear_model import LinearRegression
model_multiple = LinearRegression(fit_intercept=True)

# step 3
features_multiple = df.loc[:, ['house_age', 'metro_distance']]
target = df.loc[:, 'price_per_m2']

model_multiple.fit(features_multiple, target)
print(model_multiple.coef_)
print(model_multiple.intercept_)

from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
target_pred = model_multiple.predict(features_multiple)


print(mean_squared_error(target, target_pred))
print(r2_score(target, target_pred))


import matplotlib.pyplot as plt
import seaborn as sns
fig, axs = plt.subplots( )

target_pred_multiple = model_multiple.predict(features_multiple)
residuals = target - target_pred_multiple

residuals = target - target_pred
sns.scatterplot(x=target_pred_multiple,
                y=residuals,
                ax=axs);

axs.hlines(y=0,
          xmin=target_pred.min(),
          xmax=target_pred.max(),
          color='black')
axs.set(ylabel='Residuals',xlabel='Predicted y-values');


# If this is close to zero, you can assume that there is no correlation between the features.
# If the value is 1 or -1, there is a perfect correlation and the assumption is clearly violated. 
# Generally speaking, you should be concerned if you have a value exceeding 0.9 or -0.9.

df.corr()


# What do you do if the fifth assumption of the multiple linear regression model is not fulfilled? Then you have three options:
# * *Feature selection*: if there are two correlated features, remove one. You can discuss which is the best one to remove with experts in the field where the data comes from.
# * *Feature engineering*: You can force the features to be independent with a principal component analysis.
# * Alternative model: You can use ridge or lasso regression to reduce correlated features to the most important feature and therefore make predictions.


# Congratulations:
# You have learned the five most important assumptions of the multiple linear regression model:
# 1. The data points are independent from each other
# 2. There is a linear dependency between feature and target
# 3. The residuals are normally distributed
# 4. The residuals have a constant variance
# 5. The Features are independent from each other
# You have now learned how to check the fifth assumption with a correlation value. If you use more than two features, you can proceed similarly, 
# or you can check whether the slope values of the linear multiple regression roughly match those of a ridge or lasso regression, which you'll encounter later in this chapter.

# We've now used two of six possible features. 
# Why not include all eleven features in the model? 
# Because this leads to a whole new problem: *overfitting*. You'll find out what that means in the next lesson.


# Remember
# A multiple linear regression uses more than one feature.
# Adding additional features often improves the predictions.