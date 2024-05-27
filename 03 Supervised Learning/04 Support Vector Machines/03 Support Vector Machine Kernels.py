# What a kernel is.
# Which principle the SVM follows to generate nonlinear decision boundaries.
# How you use the 'poly' and 'rbf' kernels.

# The kernel trick

# Roughly speaking, an SVM tries to solve a set of special equations in parallel. This is called the *quadratic programming problem*

# There are two possibilities for this:

# Approach 1: First you transform your data and then you deal with the quadratic programming problem.
# Approach 2: First you deal with the quadratic programming problem and transform your data in between.

# Fascinatingly, the reason for this is that transformations that you make in between (approach2) 
# are much much simpler compared to transformations at the beginning (approach 1). 
# These simple transformations are called **kernel functions** and the mathematical insight to take the second approach is called the **kernel trick**. 
# Without question it's this kernel trick that gives an SVM its real power. A linear kernel means that you do nothing with the data for the transformation.


# Curved separators - the polynomial kernel

import pandas as pd
df_nonlinear = pd.read_csv('nonlinear_data.csv')
df_nonlinear.head()

import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots( )
sns.scatterplot(x=df_nonlinear.loc[:,'feature_1'], y=df_nonlinear.loc[:,'feature_2'], hue=df_nonlinear.loc[:,'class'],ax=ax )
ax.legend(title='SVM');


# import SVC and cross_val_score
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

# split df into features and target
features = df_nonlinear.loc[:, ['feature_1', 'feature_2']]
target = df_nonlinear.loc[:, 'class']

# training and validation
model_lin = SVC(kernel='linear')  # instantiate and train linear svm

# 5-fold cross-validation using f1-score and all available cpus
scores = cross_val_score(estimator=model_lin, X=features, y=target, cv=5, scoring='f1', n_jobs=-1)  

print('Mean f1-score:', scores.mean())



# SVM to use a polynomial kernel opens up new hyperparameters.
# The first is degree and indicates the degree of the polynomial. With this information, the SVM creates new polynomial features up to the specified degree. We learned how to create polynomial features in Polynomial Regression.
# The second parameter is gamma. This parameter controls the importance (the coefficients) of the new polynomial features. A small gamma value corresponds to a small influence and a high gamma value corresponds to a large influence for the new features. The specific value gamma='scale' causes this parameter to be selected automatically.


model_poly = SVC(kernel='poly', degree=2,gamma='scale')

scores = cross_val_score(estimator=model_poly, X=features, y=target, cv=5, scoring='f1', n_jobs=-1)  
print('Mean f1-score:', scores.mean())



# The Gaussian kernel
# 'rbf' kernel = Radial Basis Function

# The corresponding transformation of the data is as follows:
# All the data points are made to "run" like drops of ink. Imagine that our scatter plot above showed drops of ink instead of data points.
# These drops of ink run, leaving most ink in the middle of the droplet and less at the edges.
# Then it's measured how much the droplets touch each other (think of two droplets next to each other, where eventually their edges touch).
# The intensity of this touch point is then a measure for the different categories of data. All possible touch points then form new features.

# 𝐍𝐮𝐦𝐛𝐞𝐫 𝐨𝐟 𝐠𝐞𝐧𝐞𝐫𝐚𝐭𝐞𝐝 𝐟𝐞𝐚𝐭𝐮𝐫𝐞𝐬=((𝐍𝐮𝐦𝐛𝐞𝐫 𝐨𝐟 𝐝𝐚𝐭𝐚 𝐩𝐨𝐢𝐧𝐭𝐬)⋅(𝐍𝐮𝐦𝐛𝐞𝐫 𝐨𝐟 𝐝𝐚𝐭𝐚 𝐩𝐨𝐢𝐧𝐭𝐬−1))/2

model_rbf = SVC(kernel='rbf' , gamma='scale')
scores = cross_val_score(estimator=model_rbf, X=features, y=target, cv=5, scoring='f1', n_jobs=-1)  
print('Mean f1-score:', scores.mean())



# Kernel adaptability
df_nonlinear_other = pd.read_csv('nonlinear_data_2.csv')

sns.scatterplot(data=df_nonlinear_other, x='feature_1', 
                y='feature_2', hue='class', 
                palette=['#3399db', '#854d9e'], s=60);

features_other = df_nonlinear_other.loc[:, ['feature_1', 'feature_2']]
target_other = df_nonlinear_other.loc[:, 'class']

model_poly = SVC(kernel='poly', degree=2,gamma='scale')
scores = cross_val_score(estimator=model_poly, X=features_other, y=target_other, cv=5, scoring='f1', n_jobs=-1)
print('Mean f1-score poly 2:', scores.mean())
model_poly = SVC(kernel='poly', degree=3,gamma='scale')
scores = cross_val_score(estimator=model_poly, X=features_other, y=target_other, cv=5, scoring='f1', n_jobs=-1)
print('Mean f1-score poly 3:', scores.mean())
model_poly = SVC(kernel='poly', degree=4,gamma='scale')
scores = cross_val_score(estimator=model_poly, X=features_other, y=target_other, cv=5, scoring='f1', n_jobs=-1)
print('Mean f1-score poly 4:', scores.mean())
model_poly = SVC(kernel='poly', degree=5,gamma='scale')
scores = cross_val_score(estimator=model_poly, X=features_other, y=target_other, cv=5, scoring='f1', n_jobs=-1)
print('Mean f1-score poly 5:', scores.mean())
scores = cross_val_score(estimator=model_rbf , X=features_other, y=target_other, cv=5, scoring='f1', n_jobs=-1)
print('Mean f1-score rbf:', scores.mean())



# Remember:

# Use polynomial kernels with kernel='poly', degree and gamma.
# Use the gaussian kernel with kernel='rbf' and gamma.
# SVMs always form decision lines, planes or hyperplanes. The kernels then decide which dimensions to work in and with which features.