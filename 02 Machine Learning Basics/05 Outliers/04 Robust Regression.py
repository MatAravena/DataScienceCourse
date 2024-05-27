# The advantages of robust regression will become clear by comparing it with independent test data without outliers.
# **use a RANSAC regression.

# hydro_data.p (arithmetic means and standard deviations) 
# robust_hydro_data.p (medians and median absolute deviations)
import pandas as pd 
df_train = pd.read_pickle('hydro_data.p')
df_robust_train = pd.read_pickle('robust_hydro_data.p')


import matplotlib.pyplot as plt

# initialise figure and axes
fig, axs = plt.subplots(nrows=2, figsize=[10, 10])

# draw histograms
df_train.loc[:, 'cool_power_central'].plot(kind='hist', bins=20, ax=axs[0])
df_robust_train.loc[:, 'cool_power_central'].plot(kind='hist', bins=20, ax=axs[1])

# optimise histograms
axs[0].set(xlabel='Cooling power [kW]',
           ylabel='Frequency', 
           title='Cooling power (non-robust)'
          )
axs[1].set(xlabel='Cooling power [kW]',
           ylabel='Frequency', 
           title='Cooling power (robust)'
          )



from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

model_ols = LinearRegression()

# create features matrix and target vector
features_train = df_train.loc[:, 'temp_1_central':'temp_4_dispersion']
target_train = df_train.loc[:, 'cool_power_central']

# cross validation
cv_results_ols = cross_val_score(estimator=model_ols,
                                 X=features_train,
                                 y=target_train,
                                 cv=5,
                                 scoring='neg_mean_squared_error')

cv_results_ols.mean()
# -0.7870700613243057
# You may wonder why we ended up with a negative mean squared error (MSE) here now - this is because sklearn conventionally always 
# tries to maximize its score, so loss functions like MSE have to be negated here (zero is greater than any negative number). 
# If we get a negative MSE value in sklearn like we did here, that's not a bad thing, because we are only interested in the absolute MSE value - 
# which is therefore always positive. However, it's important to understand that the MSE is a quadratic measure, 
# i.e. you always have to take its square root to determine the prediction error.


# create features matrix and target vector
features_robust_train = df_robust_train.loc[:, 'temp_1_central':'temp_4_dispersion']
target_robust_train = df_robust_train.loc[:, 'cool_power_central']

# cross validation
cv_results_robust_ols = cross_val_score(estimator=model_ols,
                                        X=features_robust_train,
                                        y=target_robust_train,
                                        cv=5,
                                        scoring='neg_mean_squared_error')

# summarise scores
cv_results_robust_ols.mean()
# -0.34117095732490893


# Predicting test data without outliers

# They contain summary indicators for each test cycle: hydro_test.csv contains mean values and standard deviations, 
# while hydro_robust_test.csv contains median values and median absolute deviations
df_test = pd.read_csv('hydro_test.csv')
df_robust_test = pd.read_csv('hydro_robust_test.csv')
df_robust_test.head()


# MSE with values which are not robust
from sklearn.metrics import mean_squared_error

# model fitting
model_ols.fit(features_train, target_train)

# create features matrix
features_test = df_test.loc[:, 'temp_1_central':'temp_4_dispersion']

# predict new target values
target_test_pred_ols = model_ols.predict(features_test)

# save true target values
target_test = df_test.loc[:,'cool_power_central']

# compare actual and predicted target values
mean_squared_error(target_test, target_test_pred_ols)
# 4.504564052960768

# MSE with values which are robust
# model fitting
model_ols.fit(features_robust_train , target_robust_train)

# create features matrix
features_robust_test = df_robust_test.loc[:, 'temp_1_central':'temp_4_dispersion']

# predict new target values
target_test_pred_ols = model_ols.predict(features_robust_test)

# save true target values
target_robust_test = df_robust_test.loc[:,'cool_power_central']

# compare actual and predicted target values
mean_squared_error(target_robust_test, target_test_pred_ols)
# 13.789232288315805

# **Important:** This problem was not apparent during cross validation. 
# The reason is obvious: For cross validation, the model quality measures are calculated with a validation set, which itself can also contain outliers.
# So a high model quality according to cross validation does not protect against outliers influencing the predictions too much. 
# You should therefore take a good look at your data set in advance and have a test data set without outliers up your sleeve.




# # Predictions with RANSAC regression

# RANSAC stands for RANdom SAmple Consensus.
# As the name suggests, the point is that random samples only lead to similar model values if they don't contain outliers.
# RANSAC regression therefore assumes that the non-outliers, or inliers of the data set lead to the model learning approximately the same parameters,
# while the outliers produce all kinds of extreme deviations without consensus.

# RANSAC regression proceeds as follows:

# 1 - Select a random sample of data points. The default sample size is one larger than the number of features. All these data points are considered to be hypothetical inliers.
# 2 - Fit a linear regression model to the data.
# 3 - All the data points from the entire dataset, whose distance to their prediction is too large, are labeled as outliers.
    # All other data points of the dataset are considered to be inliers.
    # From what point is the distance too large and when does the data point become an outlier?
    # The median absolute deviation of the residuals (distances from the predicted value to the actual observed value) is used by default for this purpose.
# 4 - The inliers are included in the sample. Then points 3 and 4 are repeated.
# 5 - If no more inliers are expected, the model is only trained only with the inliers in the sample. The outliers are completely ignored.

from sklearn.linear_model import RANSACRegressor
# RANSACRegressor(min_samples=int, #Minimum number of samples chosen at random from original data
#                 residual_threshold=float, #Maximum residual for a data sample to be classified as an inlier (MAD by default)
#                 max_trials=int #Maximum number of iterations for random sample selection )

model_ransac = RANSACRegressor()
model_ransac.fit(features_train, target_train)


target_test_pred_ransac = model_ransac.predict(features_test)
mean_squared_error(target_test, target_test_pred_ransac)
# 0.11577041343874139

model_ransac.fit(features_robust_train, target_robust_train)

target_robust_test_pred_ransac = model_ransac.predict(features_robust_test)
mean_squared_error(target_robust_test, target_robust_test_pred_ransac)
# 0.00043099519286035224

# Remember:

# RANSACRegressor is a robust equivalent to LinearRegression
# Robust methods can only handle a small number of outliers. 
# If the proportion of outliers in the data set becomes too large, robust procedures can be stretched to their limits.