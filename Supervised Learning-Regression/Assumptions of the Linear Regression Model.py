import seaborn as sns
df_anscombe = sns.load_dataset("anscombe", cache=False)
sns.lmplot(x="x",
           y="y",
           col="dataset",
           data=df_anscombe,
           ci=None,
           height=4)

from sklearn.linear_model import LinearRegression
model = LinearRegression(fit_intercept=True)

for category in df_anscombe.loc[:, 'dataset'].unique():
    mask = df_anscombe.loc[:, 'dataset'] == category

    features = df_anscombe_cat = df_anscombe.loc[mask, ['x']]
    target = df_anscombe_cat = df_anscombe.loc[mask, 'y']
    model.fit( features, target)

    print(category)
    print(model.coef_)
    print(model.intercept_)


from sklearn.metrics import mean_squared_error,r2_score

for category in df_anscombe.loc[:, 'dataset'].unique():
    mask = df_anscombe.loc[:, 'dataset'] == category

    features =  df_anscombe.loc[mask, ['x']]
    target = df_anscombe.loc[mask, 'y']
    model.fit(features, target)

    target_pred = model.predict(features)
    # target_pred = [  1, 2,3,4,5,6,7,8,9,10,11 ]

    print(category)
    print(round( mean_squared_error(target, target_pred),4))
    print(round(r2_score(target, target_pred),3))



# Residuals
# residual is the vertical distance between the regression line and the data point.

# extract data for first quarter of anscombe's quartet
mask = df_anscombe.loc[:, 'dataset']=='I'
df_anscombe_cat = df_anscombe.loc[mask, :]

# choose model
from sklearn.linear_model import LinearRegression

# instantiate model
model_cat = LinearRegression()

# feature matrix and target vector
features = df_anscombe_cat.loc[:, ['x']]
target = df_anscombe_cat.loc[:, 'y']

# model fitting
model_cat.fit(features, target)

# model predictions
target_pred = model_cat.predict(features)
residuals = target - target_pred



import matplotlib.pyplot as plt
import seaborn as sns

fig, axs = plt.subplots(ncols=4, figsize=[16, 5])
model_cat = LinearRegression()

categories = {
    'I' : 0,
    'II' : 1,
    'III' : 2,
    'IV' : 3,   
}

for category in df_anscombe.loc[:, 'dataset'].unique():
    mask = df_anscombe.loc[:, 'dataset'] == category

    features = df_anscombe.loc[mask, ['x']]
    target = df_anscombe.loc[mask, 'y']

    # model fitting
    model_cat.fit(features, target)

    # model predictions
    target_pred = model_cat.predict(features)
    residuals = target - target_pred

    numericCateory = categories[category]
    sns.scatterplot(x=target_pred,
                    y=residuals,
                    ax=axs[categories[category]])

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



# Remember:
# Residuals are the distance between prediction and reality
# A residual plot shows the residuals in connection with the predictions.
# If you do not see a random scattering in the residual plot, you should use an alternative model.
