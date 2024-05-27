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
model_stores = LinearRegression(fit_intercept=True)

features = df.loc[:, ['number_convenience_stores']]
target = df.loc[:,'price_per_m2']
model_stores.fit( features, target)
print(model_stores.intercept_)
print(model_stores.coef_)


from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
target_pred = model_stores.predict(features)



# 𝑅𝑀𝑆𝐸= √𝑀𝑆𝐸
# 𝑀𝑆𝐸  = mean_squared_error(target, target_pred)
# 𝑅𝑀𝑆𝐸  = np.sqrt(mean_squared_error(target, target_pred))
print(mean_squared_error(target, target_pred))
print(np.sqrt(mean_squared_error(target, target_pred)))

# Dispersion
print(r2_score(target, target_pred))



import matplotlib.pyplot as plt
import seaborn as sns

model = LinearRegression(fit_intercept=True)
fig, axs = plt.subplots(ncols=3, figsize=[16, 5])

categories = {
    'house_age' : 0,
    'metro_distance' : 1,
    'number_convenience_stores' : 2,
}


for category in ['house_age', 'metro_distance', 'number_convenience_stores']:
    features = df.loc[:, [category]]

    # model fitting
    model.fit(features, target)

    # model predictions
    target_pred = model.predict(features)
    residuals = target - target_pred

    numericCateory = categories[category]

    sns.scatterplot(x=target_pred,
                    y=residuals,
                    ax=axs[numericCateory])

    # labels
    axs[numericCateory].set(xlabel='Predicted y-values',
               ylabel='Residuals',
               title='Dataset:{0}'.format(category))

    # zero line
    axs[numericCateory].hlines(y=0,
              xmin=target_pred.min(),
              xmax=target_pred.max(),
              color='black')

fig.tight_layout()