# Why linear regression is unsuitable for classification tasks.
# What logistic regression is and how it makes predictions.
# How to use logistic regression in sklearn.


# Logistic regression is a well-known algorithm for classification tasks. Although it's mentioned in the name, this algorithm can't be used for regression tasks.

import pandas as pd
df = pd.read_csv('social_media_train.csv', index_col=0)
df.head()

# The problem with linear regression
from sklearn.linear_model import LinearRegression
model_linear =  LinearRegression()
features     = df.loc[:,['ratio_numlen_username']]
target       = df.loc[:,'fake']
model_linear.fit(features, target)

intercept = model_linear.intercept_
slope = model_linear.coef_

df.describe()

intercept + (slope * features.loc[:, 'ratio_numlen_username'].mean())
intercept + (slope * 0.8)

# import modules
import seaborn as sns
import matplotlib.pyplot as plt

#matplotlib style sheet
plt.style.use('fivethirtyeight')

# initialize figure and axes
fig, ax = plt.subplots(figsize=[10, 5])

# scatter plot with regression line
sns.regplot(x="ratio_numlen_username",
            y="fake",
            data=df,
            ax=ax,
            scatter_kws={'alpha': 0.2},
            ci=False)

# orange vertical line
ax.axvline(0.53, 
           ls='--', 
           color = "orange")

# labels
ax.set_title("Linear Regression")
ax.set_xlabel("Ratio of numbers in username to its length")
ax.set_ylabel("Probability of being fake user")
ax.set_xlim([0, 0.85])


# Logistic Regression - A Continuous Classifier

# Logistic regression calculates a prediction probability,  𝐏𝐫𝐨𝐛𝐚𝐛𝐢𝐥𝐢𝐭𝐲𝐥𝐨𝐠𝐑𝐞𝐠,
# by running the prediction probability of the linear regression,  𝐏𝐫𝐨𝐛𝐚𝐛𝐢𝐥𝐢𝐭𝐲𝐥𝐢𝐧𝐑𝐞𝐠,
# through a sigmoid function. In plain language this means:




# The object  𝑒
#   is called Euler's number and its value is roughly 2.71

# The Sigmoid function is that it only outputs continuous values between 0 and 1. 

# Making predictions with logistic regression
from sklearn.linear_model import LogisticRegression 

#  LogisticRegression(  C=float,        #regularization strength: smaller values means higher regularization
#                       penalty=str,    #type of regularization: "l1" for Lasso, "l2" for Ridge
#                       solver=str)     #learning strategy      

# For small data sets solver='liblinear' and for large datasets solver='sag'.
# If you want to do Lasso regularization, only 'liblinear' or 'saga' is possible. With ridge regularization you can choose between 'saga', 'lbfgs' or 'newton_cg'.
# solver by default  is 'lbfgs'

model_log = LogisticRegression()
model_log.fit(features, target)

import math
intercept = model_log.intercept_
slope     = model_log.coef_
e         = math.e 

mean = features.loc[:, 'ratio_numlen_username'].mean()
1 / (1 + e**(-(intercept + (slope * mean))))

# We end up with an average probability of 54% that an account is fake.

# initialize figure with two axes in two rows
fig, axs = plt.subplots(nrows=2, figsize=[10, 12])

# plot the linear regression line
sns.regplot(x="ratio_numlen_username", 
            y="fake", 
            data=df, 
            ax=axs[0],
            scatter_kws={'alpha': 0.2},
            ci=False)

# add labels
axs[0].set(title="Linear Regression", 
           xlabel="Ratio of numbers in username to its length", 
           ylabel="Probability of fake account", 
           ylim=[-0.05, 1.4],
           xlim=[0.0, 0.85])

# plot the logistic regression line
sns.regplot(x="ratio_numlen_username", 
            y="fake", 
            data=df, 
            ax=axs[1], 
            scatter_kws={'alpha': 0.2},
            logistic=True,
            ci=False)

# add labels
axs[1].set(title="Logistic Regression", 
           xlabel="Ratio of numbers in username to its length", 
           ylabel="Probability of fake account", 
           ylim=[-0.05, 1.4],
           xlim=[0.0, 0.85]);

# logistic regression's sigmoid function always predicts positive values below 1.0.



# Deep dive: At this point, you might want to know how exactly a solver finds an optimal solution. 
# The answer goes right to the heart of how machine learning works. At this point we only want to scratch the surface of the concept, 
# because the correct treatment would require some prior knowledge about "Analysis on manifolds" and "Numerical Optimization". 
# This would be far too much to cover in this course and we want to take a heuristic approach.

# All solvers use the same method, which is called gradient descent (gradient method). Here the basis is a function that measures the
# error between real values and model predictions. You call this function cost function - and each machine learning model has its 
# own cost function. If the cost function outputs the value 0, the model is perfect and all predictions are perfect.

# However, perfection is never attainable in the real world, since the real values are never perfect (this is what we call noise, which is extremely
# difficult for a model to understand and should usually not be there). The aim of the solver is therefore to bring the 
# cost function as close as possible to 0 - we are therefore approximating the zeroes. As soon as the solver believes it is close enough, 
# it has found the optimal solution. We say that the solver has converged. The only thing the solver needs to know to converge is 
# where to start and how big the steps should be to get to the 0. The starting point is guessed and different solvers have different methods for this.

# The size of the steps is also individual for each solver and is controlled by the first derivative of the cost function, which is also
# called the gradient (hence the name gradient descent).



# Remember:

# Use linear regression for continuous target values
# Logistic regression for the probabilities of target categories (classification)
# solvers (also called optimizers) are algorithms that can help minimize the cost function.

# Literature:
# Sigmoid function
# Euler's number
# Gradient descent