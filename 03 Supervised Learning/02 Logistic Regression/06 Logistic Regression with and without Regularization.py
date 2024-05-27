# The C parameter of LogisticRegression() for specifying regularization
# my_model.predict_proba() to predict probabilities.


# Logistic regression without regularization

# module import
import pandas as pd
import pdpipe as pdp

# data read in
df_train = pd.read_csv("social_media_train.csv", index_col=[0])

# label encoding
dict_label_encoding = {'Yes': 1, 'No': 0}
df_train.loc[:, 'profile_pic'] = df_train.loc[:, 'profile_pic'].replace(dict_label_encoding)
df_train.loc[:, 'extern_url'] = df_train.loc[:, 'extern_url'].replace(dict_label_encoding)
df_train.loc[:, 'private'] = df_train.loc[:, 'private'].replace(dict_label_encoding)

# one-hot encoding
onehot = pdp.OneHotEncode(["sim_name_username"], drop_first=False)
df_train = onehot.fit_transform(df_train) #fit and transform to training set

# look at data
df_train.head()


# By default, the logistic regression algorithm of sklearn already uses regularization - with the regularization parameter C = 1.0 by default.
# If we assign an extremely large value to C, such as a 1 followed by 42 zeros (1e42), it doesn't perform any regularization.
# That is what we want to achieve here first.


# 1. Model specification
from sklearn.linear_model import LogisticRegression

# 2. Model instantiation
model_log = LogisticRegression(solver='lbfgs', max_iter=1e4, C=1e42, random_state=42)

# 3. Features matrix and target vector
features_train = df_train.iloc[:, 1:]
target_train = df_train.loc[:, 'fake']

# 4. Model fitting
model_log.fit(features_train, target_train)


 
from sklearn.model_selection import cross_val_score 
cv_results = cross_val_score(estimator=model_log,
                                 X=features_train,
                                 y=target_train,
                                 cv=5,
                                 scoring='accuracy')
cv_results.mean()
# score 0.9252923538230885

# read in data
df_aim = pd.read_csv("social_media_aim.csv", index_col=[0])

# label encoding
dict_label_encoding = {'Yes': 1, 'No': 0}
df_aim.loc[:, 'profile_pic'] = df_aim.loc[:, 'profile_pic'].replace(dict_label_encoding)
df_aim.loc[:, 'extern_url'] = df_aim.loc[:, 'extern_url'].replace(dict_label_encoding)
df_aim.loc[:, 'private'] = df_aim.loc[:, 'private'].replace(dict_label_encoding)

# one-hot encoding
df_aim = onehot.transform(df_aim) #transform to target set

# Look at data
df_aim.head()

features_aim = df_aim.copy()
df_aim.loc[:, 'fake_pred_log'] = model_log.predict(features_aim)
df_aim




# Logistic regression with regularization
# To understand how regularization works in logistic regression, it is worth looking back at ridge regression. In the lesson Regularization (Module 1 Chapter 1) you learned that a linear regression with regularization has two goals. For a ridge regression, they would be something like this:

# Keep the difference between predicted and actual target values as small as possible.
# Keep the sum of the squared slopes (e.g.  (𝑠𝑙𝑜𝑝𝑒1)2+(𝑠𝑙𝑜𝑝𝑒2)2
#  ) as small as possible.
# The second objective is called regularization, or shrinkage penalty. This means that the model would be punished if the slopes are too big. The alpha parameter of Ridge() controls how much the second goal should be pursued.

# With alpha=0 the second objective (regularization) is ignored. Then the ridge regression would be a normal linear regression. With an infinitely high alpha, the first objective is disregarded. In this case all slopes are zero.

# In the lesson Optional: Logistic Regression as Linear Regression with Log Odds you saw that logistic regression is a kind of alternative version of linear regression. So you can also think of logistic regression with regularization as a ridge regression that predicts log chances.

# Let's start by instantiating the model. The C parameter of LogisticRegression() controls the balance between the top two targets, similar to the alpha parameter of Ridge(). Although they have the same function - confusingly, they have the exact opposite effect. The following settings are equivalent:

# Description	            alpha	C
# no regularization	        0	    Inf
# little regularization	    0.5	    2
# Standard regularization	1	    1
# a lot of regularization	0.2	    5
# maximum regularization	Inf	    0


model_reg = LogisticRegression(solver='lbfgs', C=0.5, max_iter=1e4,random_state=42)


# So C and alpha are related to each other in the following way: C = 1/alpha. At the end you have to find the optimal value for C again with a grid search. In this case we'll try 0.5. Instantiate LogisticRegression and store the model in the variable model_reg. Use the following hyperparameter settings: solver='lbfgs', C=0.5 and max_iter=1e4. To be able to reproduce this model, you should also specify random_state=42.

# Attention: In contrast to the logistic regression without regularization, the features of the model with regularization definitely have to be standardized! This is due to the penalty parameter (l2 by default) in logistic regression: as with linear regression, logistic regression with regularization makes the prediction dependent on feature scaling, with  𝐿1
#   (lasso) and  𝐿2
#   (ridge) penalizing large coefficients more heavily. In order for the coefficients to be penalized equally, we have to standardize them.

# You should always consider which scaling method is better for your data set(s). However, for logistic regression with regularization, the Z-transformation (StandardScaler) is recommended for the majority of cases, see also the references at the end of this lesson.




from sklearn.preprocessing import StandardScaler #use StandardScaler to adjust the features

scaler = StandardScaler()
features_train_scaled = scaler.fit_transform(features_train)  # fit to training data and scale it
features_aim_scaled = scaler.transform(features_aim)  #scale target data

features_train_scaled = pd.DataFrame(features_train_scaled, columns=features_train.columns)

model_reg.fit(features_train_scaled,target_train)

df_aim.loc[:,'fake_pred_reg'] =  model_reg.predict(features_aim_scaled)
df_aim



# Predicted probabilities

# If you train a classification model, you always predict the categories with my_model.predict().
# But if you want to know the probability of something belonging to one category or another, you should use my_model.predict_proba().
# It's probably best to try that out now.
target_aim_pred_proba = model_log.predict_proba(features_aim)
target_aim_pred_proba

# suppress scientific notation in numpy
import numpy as np  # import module
np.set_printoptions(suppress=True)  # suppress scientific notation

# suppress scientific notation in pandas
pd.options.display.float_format = '{:.2f}'.format

target_aim_pred_proba  # print array


df_aim.loc[:, 'fake_pred_log']

df_aim.loc[:, 'fake_pred_log_proba'] = target_aim_pred_proba[:, 1]
df_aim


df_aim.loc[:, 'fake_pred_reg_proba'] = model_reg.predict_proba(features_aim_scaled)[:, 1]
df_aim

# By default, LogisticRegression uses 0.5 as the threshold value. 
# For example, in our case if the probability of an account being fake is less than 0.5 (50%), 
# it is predicted that the account is not fake. If it is higher, it is predicted to be fake.


# Remember:

# LogisticRegression uses regularization as standard.
# While it's not necessary to standardize features for the logistic regression model without regularization, it is always necessary for a logistic regression model with regularization! A Z-transformation of the features using StandardScaler is suitable for most cases.
# The higher the C parameter of LogisticRegression(), the weaker the regularization.
# With my_model.predict_proba() predict probabilities.
