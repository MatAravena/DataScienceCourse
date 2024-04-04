import pandas as pd
import numpy as np
df_train = pd.read_excel('Taiwan_real_estate_training_data.xlsx', index_col='No')
df_test = pd.read_excel('Taiwan_real_estate_test_data.xlsx', index_col='No')
df_aim = pd.read_excel('Taiwan_real_estate_prediction_data.xlsx', index_col='No')

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
df_train.columns = col_names
df_test.columns = col_names
df_aim.columns = col_names

df_train.loc[:, 'price_per_m2'] = df_train.loc[:, 'price_per_ping'] / 3.3
df_test.loc[:, 'price_per_m2'] = df_test.loc[:, 'price_per_ping'] / 3.3
df_aim.loc[:, 'price_per_m2'] = df_aim.loc[:, 'price_per_ping'] / 3.3

df_train = df_train.drop('price_per_ping', axis=1)
df_test = df_test.drop('price_per_ping', axis=1)
df_aim = df_aim.drop('price_per_ping', axis=1)


# Ridge regression, 
# also known as Tikhonov regularization uses regularization to avoid overfitting. When the model is fitted to the data, there are two objectives that should be pursued:

# Keep the difference between predicted and actual target values as small as possible.
# Keep the sum of the squared slopes (e.g.  (𝑠𝑙𝑜𝑝𝑒1)2+(𝑠𝑙𝑜𝑝𝑒2)2 ) as small as possible.



from sklearn.linear_model import Ridge


# Ridge(
#     alpha= float,       #strength of penalty for regularization
#     fit_intercept=True, #fit intercept in underlying linear regression
#     solver='auto',      #solving algorithm, will affect training runtime
#     random_state=None,  #random seed used for data shuffling
# )



model_ridge = Ridge(1500)

features_train = df_train.loc[:, col_names[:-1]]
target_train = df_train.loc[:, 'price_per_m2']

features_train.describe()

# How can you tell the ridge regression to treat all features the same? You can use a trick. You standardize the values of each feature. 
# This means that from each value in features ( 𝑥 ) the mean of its own column ( 𝑥¯ ) is subtracted and each value is divided by the standard deviation ( 𝜎 ) of its column. 

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

#ignore DataConversionWarning
from sklearn.exceptions import DataConversionWarning
import warnings
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

# Standarized  --> test results
features_train_standardized = scaler.fit_transform(features_train)


# all columns in features_train_standardized should be zero, while the standard deviations (a measure for the dispersion of the data in the row 'std') in each column should be 1.
pd.options.display.float_format = '{:.2f}'.format  # avoid scientific notation using exponent, display up to two digital places instead
pd.DataFrame(features_train_standardized).describe()  # eight value summary


features_train_standardized = pd.DataFrame(features_train_standardized)
features_train_standardized.columns =  features_train.columns


model_ridge.fit(features_train_standardized, target_train)



import matplotlib.pyplot as plt

fig, ax = plt.subplots(1,figsize=(6,6))
pd.Series(model_ridge.coef_, index=features_train.columns).sort_values().plot.bar(ax=ax)
ax.set_title('Feature Weights in Ridgemodel')
ax.set_ylabel('Weight (standardized)')
ax.set_xlabel('Feature')
fig.tight_layout()



# *********************
# Determing the ridge regression model quality

features_test  = df_test.loc[:, col_names[:-1]]
target_test = df_test.loc[:, 'price_per_m2']

features_test_standardized = scaler.transform(features_test)
target_test_pred_ridge = model_ridge.predict(features_test_standardized)

from sklearn.metrics import mean_squared_error, r2_score
print(mean_squared_error(target_test , target_test_pred_ridge ))
print(r2_score(target_test , target_test_pred_ridge ))

# Remember:
# Ridge minimizes the sum of the squared slope values
# Ridge is suitable to prevent overfitting and colinearity
# Only ever fit to the training set!