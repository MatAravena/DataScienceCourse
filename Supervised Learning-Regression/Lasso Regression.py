import pandas as pd
import numpy as np
from sklearn.metrics import *
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
              'longtitude', 
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

features_train = df_train.drop('price_per_m2', axis=1)
features_test = df_test.drop('price_per_m2', axis=1)
target_train = df_train.loc[:,'price_per_m2']
target_test = df_test.loc[:,'price_per_m2']




# Fit the model to the data with all eleven features. Note that the features should be standardized, as with the ridge regression. Print the slopes at the end
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(features_train)
features_train_standardized = scaler.transform(features_train) 

model_lasso = Lasso(alpha=1)
model_lasso.fit(features_train, target_train)
model_lasso.fit(features_train_standardized, target_train)

print(model_lasso.coef_)
print(features_train.columns)






from sklearn.metrics import mean_squared_error, r2_score
 
features_test_standardized = scaler.transform(features_test)
target_test_pred_lasso = model_lasso.predict(features_test_standardized)

print('MSE: ', mean_squared_error(target_test, target_test_pred_lasso))
print('RMSE: ', np.sqrt(mean_squared_error(target_test, target_test_pred_lasso)))
print('R2: ', r2_score(target_test, target_test_pred_lasso))








df_aim =  df_aim.dropna(axis=0, how='all')

features_aim = df_aim.drop('price_per_m2', axis=1)
target_aim = df_aim.loc[:,'price_per_m2']

features_aim_standardized = scaler.transform(features_aim)
target_aim_pred_lasso = model_lasso.predict(features_aim_standardized)
print(target_aim_pred_lasso.mean())

#print('MSE: ', mean_squared_error(target_aim, target_aim_pred_lasso))
#print('RMSE: ', np.sqrt(mean_squared_error(target_aim, target_aim_pred_lasso)))
#print('R2: ', r2_score(target_aim, target_aim_pred_lasso))



# Remember:

# Lasso minimizes the sum of the absolute slope values.
# Lasso is suitable for preventing overfitting and for feature selection.