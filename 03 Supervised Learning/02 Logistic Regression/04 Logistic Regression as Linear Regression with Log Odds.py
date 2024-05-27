# module import
import pandas as pd

# data gathering
df = pd.read_csv("social_media_train.csv", index_col=[0])

#look at data
df.head()


# Linear regression and Logistic regression basically use the same formula. The only difference is that the linear regression predicts target values with the formula, 
# while logistic regression uses it to predict **log odds**. To understand this, we first need to look at what **log odds** actually are.


probability_sunny = 4 / 7 
print("Probability of sunny day:", round(probability_sunny, 2))



# odds = probability / ConverseProbaility

probability_not_sunny = 1 - probability_sunny
print("Probability of *not* sunny day:", round(probability_not_sunny, 2))

odds_sunny= probability_sunny / probability_not_sunny
print("The odds of sunny day:", round(odds_sunny, 2))



probability_sunny = 3 / 7
probability_not_sunny = 1 - probability_sunny
odds_sunny= probability_sunny / probability_not_sunny
print("The odds of sunny day:", round(odds_sunny, 2))

import numpy as np
print("Log odds of having sunny day:", round(np.log(odds_sunny), 2))

# Rainy day
print("Log odds of having sunny day:", round(np.log(probability_not_sunny /probability_sunny), 2))

# a positive log odds value means that the odds are in favor of the case arising

#     x                                   probability of case arising   odds of case	      log odds of the case
# probability > converse-probability	    𝑝𝑟𝑜𝑏𝑎𝑏𝑖𝑙𝑖𝑡𝑦>0.5                 𝑜𝑑𝑑𝑠>1              𝑙𝑜𝑔(𝑜𝑑𝑑𝑠)>0
        
# probability == converse-probability	    𝑝𝑟𝑜𝑏𝑎𝑏𝑖𝑙𝑖𝑡𝑦=0.5                 𝑜𝑑𝑑𝑠=1              𝑙𝑜𝑔(𝑜𝑑𝑑𝑠)=0
        
# probability < converse-probability	    𝑝𝑟𝑜𝑏𝑎𝑏𝑖𝑙𝑖𝑡𝑦<0.5                 𝑜𝑑𝑑𝑠<1              𝑙𝑜𝑔(𝑜𝑑𝑑𝑠)<0


# You can only use a linear regression to predict categories when the probabilities have been transformed into log odds. 
# So we call linear regression with log odds **logistic regression**. It does not directly predict categories (`1` or `0`), but log odds




# Curve and line
from sklearn.linear_model import LogisticRegression
model_log = LogisticRegression(solver='lbfgs')
features_train = df.loc[:,['ratio_numlen_username']]
target_train = df.loc[:,'fake']
model_log.fit(features_train, target_train)


# create artificial feature values
features_pred = pd.DataFrame({'ratio_numlen_username': np.linspace(-0.7, 1, 32)})

# predict probabilities of artificial feature values
target_train_pred_proba = model_log.predict_proba(features_pred)

# module import for visualisation
import seaborn as sns

# line plot
ax = sns.lineplot(x=features_pred.iloc[:, 0],
            y=target_train_pred_proba[:, 1])

# labels
ax.set(title='The typical S-curve of logistic regression',
       xlabel='Feature value',
       ylabel='Probability of being in reference category')

# orange horizontal line
ax.axhline(0.5, ls='--', color = "orange")




# calculate odds
target_train_pred_odds = target_train_pred_proba[:, 1] / (1 - target_train_pred_proba[:, 1])

# line plot
ax = sns.lineplot(x=features_pred.iloc[:, 0],
            y=target_train_pred_odds)

# labels
ax.set(title='Visualising odds',
       xlabel='Feature value',
       ylabel='Odds of being in reference category')

# orange horizontal line
ax.axhline(1, ls='--', color = "orange")


# calculate log odds
target_train_pred_log_odds = np.log(target_train_pred_odds)

# line plot
ax = sns.lineplot(x=features_pred.iloc[:, 0],
            y=target_train_pred_log_odds)

# labels
ax.set(title='The typical regression line of log odds',
       xlabel='Feature value',
       ylabel='Log odds of being in reference category')

# orange horizontal line
ax.axhline(0, ls='--', color = "orange")



# This line is a regression line as you learned in the lesson Simple Linear Regression with sklearn (Module 1, Chapter 1). The special thing about it is the nature of the predicted values: Log odds So you now see log odds on the y-axis.

# You saw above that the value zero is critical for log odds. So it's displayed here as a dotted line. The categorical prediction (e.g. fake or not fake) is based on whether the probability value is below 0 or not.

# Remember this regression line of log odds if you don't know how to interpret a certain aspect of logistic regression. For example, the feature coefficients that the model learns from the training data represent slope values of this regression line of log odds. In Simple Linear Regression with sklearn (Module 1, Chapter 1) you learned that a change of 1 in the feature results in a change of the slope value in the predicted target value. In this case the target value is log odds and the principle still applies.

# Negative feature coefficients therefore represent a negative influence of the feature on the probability of predicting the feature category. By the same token, the same applies to positive feature coefficients. A feature coefficient of about 0 means that the feature has virtually no effect on the predictions.

