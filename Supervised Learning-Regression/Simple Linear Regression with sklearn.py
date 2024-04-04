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

df.loc[:,'price_per_m2'] = df.loc[:,'price_per_ping']/3.3
df.head()

from sklearn.linear_model import LinearRegression

model = LinearRegression(fit_intercept=True)


#Organize data into a feature matrix and target vector

features = df.loc[:,'house_age']
print(type(df_house_age))       # <---- Serie

# The feature matrix
features = df.loc[:,['house_age']]
print(type(df_house_age))       # <---- DataFrame


features.shape 
# The target 'vector'
target = df.loc[:,'price_per_m2']

import seaborn as sns
sns.regplot(x=features.iloc[:, 0],
            y=target,
            scatter_kws={'color':'#17415f',  # dark blue dots
                        'alpha':1},  # no transparency for dots
            fit_reg=False);  # no regression line

# alternative
#import matplotlib.pyplot as plt
#fig, ax = plt.subplots()
#ax.scatter(x=features.iloc[:, 0], 
#            y=target,
#          c='#17415f')  # dark blue dots
#ax.set(xlabel='House age (years)',
#      ylabel='House price (Dollars per square meter)')


#Model fitting
model.fit(X=Dataframe,       #features Dataframe
          y=Series,          #target Series
          sample_weight=list #*optional* individual weights for each sample
         )
model.fit(features, target)

sns.regplot(x=features.iloc[:, 0], 
            y=target,
            scatter_kws={'color':'#17415f',  # dark blue dots
                        'alpha':1},  # no transparency for dots
            line_kws={'color':'#70b8e5'},  # light blue regression line
            ci=None);  # no confidence intervals around line

# In the sklearn module, the model parameters are always found as an attribute of the model variable. 
# The attributes that end in an underscore, represent the parameters learned by the model. For example, my_model.coef_ is the slope. Print this parameter.
print(model.coef_)
print(model.intercept_)


import matplotlib.pyplot as plt

fig, ax = plt.subplots()

ax.set(xlim=[0, max(df.loc[:, 'house_age'])],  # limits of x-axis
       ylim=[8, max(df.loc[:, 'price_per_m2'])])  # limits of y-axis

sns.regplot(x=features.iloc[:, 0], 
            y=target,
            scatter=False,  # no dots
            line_kws={'color':'#70b8e5'},  # light blue regression line
            ci=None,  # no confidence intervals around line
            ax=ax)  # draw on Axes ax

# Make predictions with the trained model
features_aim = pd.DataFrame({'house_age': [0, 0, 12, 13.3, 34, 3, 15, 27.5, 11, 7]})
features_aim


# Predict
target_aim_pred = model.predict(features_aim)
target_aim_pred
fig, ax = plt.subplots()

sns.regplot(x=features.iloc[:, 0],  # house age in training data set
            y=target,  # prices in training data set
            scatter_kws={'color':'#17415f',  # dark blue dots
                        'alpha':1},  # no transparency for dots
            line_kws={'color':'#70b8e5'},  # light blue regression line
            ci=None,  # no confidence intervals around line
           ax=ax)  # plot on current Axes

sns.regplot(x=features_aim.iloc[:, 0],  # x-values of houses with estimated prices
            y=target_aim_pred,  # estimated prices
            scatter_kws={'color':'#ff9e1c',  # orange dots
                        'alpha':1,  # no transparency for dots
                        's':70},  # dot size
            fit_reg=False,  # no additional regression line
            ci=None,  # no confidence intervals around line
           ax=ax)  # plot on current Axes

ax.set(xlabel='House age [years]',
    ylabel='Predicted house price per sqaure meter \n[Taiwan Dollar]',
      xlim=[0, max(df.loc[:, 'house_age'])],
      ylim=[0, max(df.loc[:, 'price_per_m2'])])

# Remember:

# There are five steps to making data-driven predictions with sklearn:

# Choose model type
# Instantiate the model with certain hyperparameters
# Split data into a feature matrix and target vector
# Model fitting
# Make predictions with the trained model