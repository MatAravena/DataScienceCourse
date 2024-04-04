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
model_multiple_all = LinearRegression(fit_intercept=True)

features = df.loc[:, ['house_age',
 'metro_distance', 
 'number_convenience_stores', 
 'number_parking_spaces',
 'air_pollution',
 'light_pollution',
 'noise_pollution',
 'neighborhood_quality',
 'crime_score',
 'energy_consumption',
 'longitude']]
target = df.loc[:,'price_per_m2']
model_multiple_all.fit( features, target)



from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

target_pred = model_multiple_all.predict(features)

print(mean_squared_error(target, target_pred))
print(np.sqrt(mean_squared_error(target, target_pred)))
print(r2_score(target, target_pred)*100)

