import pandas as pd

df = pd.read_excel('Taiwan_real_estate_training_data.xlsx', index_col='No')
df.head()

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

df.info()
df.isna().sum()

pd.plotting.scatter_matrix(df,  figsize=(25,25));

df.loc[:,'price_per_m2'] = df.loc[:,'price_per_ping']/3.3
df.head()
