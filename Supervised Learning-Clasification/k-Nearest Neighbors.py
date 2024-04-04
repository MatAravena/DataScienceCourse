# module import
import pandas as pd

# data gathering
df_train = pd.read_csv('occupancy_training.txt')

# turn date into DateTime
df_train.loc[:, 'date'] = pd.to_datetime(df_train.loc[:, 'date'])

# turn Occupancy into category
df_train.loc[:, 'Occupancy'] = df_train.loc[:, 'Occupancy'].astype('category')

# define new feature
df_train.loc[:, 'msm'] = (df_train.loc[:, 'date'].dt.hour * 60) + df_train.loc[:, 'date'].dt.minute





# KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.preprocessing import StandardScaler


model = KNeighborsClassifier(n_neighbors=3)

features_train = df_train.loc[:,['CO2','HumidityRatio']]
target_train= df_train.loc[:,'Occupancy']

features_train = df_train.loc[:,['CO2','HumidityRatio']]
scaler = StandardScaler()
scaler.fit(features_train)
features_train_standardized = scaler.transform(features_train)

pd.options.display.float_format = '{:.2f}'.format  # avoid scientific notation using exponent, display up to two digital places instead
pd.DataFrame(features_train_standardized, columns = ['CO2', 'HumidityRatio']).describe()

# Predicting room occupancy
model.fit(features_train_standardized, target_train)
df_aim = pd.DataFrame({'CO2':[420, 10000],
                       'HumidityRatio':[0.0038, 0.005]})

features_aim = df_aim.loc[:, ['CO2', 'HumidityRatio']]

features_aim_standardized = scaler.transform(features_aim)

target_aim_pred = model.predict(features_aim_standardized)


