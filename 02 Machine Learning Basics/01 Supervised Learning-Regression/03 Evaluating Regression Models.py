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

from sklearn.linear_model import LinearRegression
model_metro = LinearRegression(fit_intercept=True)


# feature matrix
features_metro = df.loc[:,['metro_distance']]
# vector
target = df.loc[:,'price_per_m2']  
# fit model
model_metro.fit(features_metro, target)

model_metro.coef_
round( model_metro.intercept_,2)

df_aim = pd.read_excel( 'Taiwan_real_estate_prediction_data.xlsx', index_col=False)

features_aim_metro = df_aim.loc[:,['X2 distance to the nearest MRT station']]

target_aim_pred_metro = model_metro.predict(features_aim_metro)

print(target_aim_pred_metro)
print(type(target_aim_pred_metro))

import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots()

sns.regplot(x=features_metro.iloc[:, 0],  # metro distances in training data set
            y=target,  # prices in training data set
            scatter_kws={'color':'#17415f',  # dark blue dots
                        'alpha':1},  # no transparency for dots
            line_kws={'color':'#70b8e5'},  # light blue regression line
            ci=None,  # no confidence intervals around line
           ax=ax)  # plot on current Axes

sns.regplot(x=features_aim_metro.iloc[:, 0],  # x-values of houses with estimated prices
            y=target_aim_pred_metro,  # estimated prices
            scatter_kws={'color':'#ff9e1c',  # orange dots
                        'alpha':1,  # no transparency for dots
                        's':70},  # dot size
            fit_reg=False,  # no additional regression line
            ci=None,  # no confidence intervals around line
           ax=ax)  # plot on current Axes

ax.set(xlim=[0, max(df.loc[:, 'metro_distance'])])


# Comparing predictions
model_age = LinearRegression(fit_intercept=True)

features_age = df.loc[:, ['house_age']]
model_age.fit(features_age, target)

features_aim_age = df_aim.loc[:, ['X1 house age']]
target_aim_pred_age = model_age.predict(features_aim_age)

# features_aim_age  = df_aim.loc[:,['X1 house age']]
# model_age.fit(features_aim_age, target)
# target_aim_pred_age = model_age.predict(features_aim_age)


# Compare prices
print(target_aim_pred_age)
print(target_aim_pred_metro)


fig, ax = plt.subplots()

sns.scatterplot(x=target_aim_pred_age, 
                y=target_aim_pred_metro, 
                color='#ff9e1c',  # orange dots
                s=70,   # big dots
                ax=ax)  # draw on current Axes

ax.plot([10, 20], # x-values
       [10, 20],  # y-values
       'grey')

ax.set(xlabel='Predicted house price based on house age',
       ylabel='Predicted house price based on distance to metro');


# Determining the quality of regression models with the (rooted) mean squared error
# sklearn offers you six different options to determine the quality of regressions. 
# You can find them all in the official documentation. 
# They are based on predicting data with the trained model and then comparing these predictions with the real values. 
# The difference between a prediction and the real value is called a residual. It is the basis for evaluating the model. 
# A good model predicts the values as they actually are. The residual should therefore be as small as possible

# To obtain the residuals, we treat the actual age values of the houses in the training dataset (df) 
# as if we wanted a price prediction for them and store those in target_pred_age.


features_age = df.loc[:, ['house_age']]
target_pred_age = model_age.predict(features_age)
target_pred_metro = model_metro.predict(features_age)

fig, ax = plt.subplots(1, 2, figsize=[12, 6])

# Predictions based on house age
sns.scatterplot(x=target_pred_age,
                y=target,
                color='#515151',  # grey dots
                s=70,  # big dots
                ax=ax[0]) # draw on current Axes
# diagonal
ax[0].plot([0, 50], # x-values
           [0, 50], # y-values
           'grey') # line colour

ax[0].set(xlabel='Predicted house price', 
          ylabel='Actual house price',
          title='Predictions based on house age');

# Predictions based on metro distance
sns.scatterplot(x=target_pred_metro, 
                y=target, 
                color='#515151',  # grey dots
                s=70,  # big dots
                ax=ax[1]) # draw on current Axes
# diagonal
ax[1].plot([0, 50], # x-values
           [0, 50], # y-values
           'grey') # line colour

ax[1].set(xlabel='Predicted house price',
          ylabel='Actual house price',
          title='Predictions based on metro distance');



from sklearn.metrics import mean_squared_error 


# Calculate the residual and the mean squared error
import numpy as np
residuals = target - target_pred_age
residuals_squared = residuals**2
MSE = np.mean(residuals_squared)
MSE

# OR
mean_squared_error(target, target_pred_age)


# Determining the quality of regression models with the coefficient of determination
from sklearn.metrics import r2_score 

# In addition to the rooted mean squared error, the coefficient of determination, also called  𝑅2, 
# is often used to determine model quality. The coefficient of determination indicates how much dispersion in
# the data can be explained by the linear regression model. 
# The perfect regression model has a coefficient of determination of one. 
# The worst 𝑅2 value a standard linear regression can create is zero.

r2_score(target, target_pred_age)
r2_score(target, target_pred_metro)
# Result is in %


# 𝑅𝑀𝑆𝐸= √𝑀𝑆𝐸


# Remember
# The mean squared error expresses the mean squared distance between the prediction and measured value
# ( called residual )
# The RMSE is the root of the MSE. It can be directly interpreted, since it has the same dimension as the target.
# The coefficient of determination indicates how much dispersion in the data can be explained by the linear regression model
# The coefficient of determination (R²) is a number between 0 and 1 that measures how well a statistical model predicts an outcome.
