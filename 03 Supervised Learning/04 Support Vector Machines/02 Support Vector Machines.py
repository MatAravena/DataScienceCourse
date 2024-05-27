# What the support vectors are
# How SVMs use them to make predictions
# How to interpret the C parameter of the SVM




# Support Vectors
# powerful machine learning algorithm
# SVMs can perform both classifications and regressions and they are often unbeatable in their prediction power for medium and small size data sets

import pandas as pd
df = pd.read_csv('sample_data_svm.csv')



# SVC(C=float,     # regularization parameter, controls the 'strictness' of the SVM
#     kernel=str # kernel parameter, controls the learning style of the SVM
#    )

# kernel is particularly important. This parameter tells the SVM which learning strategy to follow. So kernel directly influences the predictive power and efficiency of the SVM.


from sklearn.svm import SVC
model_svm = SVC(kernel='linear', C=10)


import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots( )
sns.scatterplot(x=df.loc[:,'feature_1'], y=df.loc[:,'feature_2'], hue=df.loc[:,'class'],ax=ax )
ax.legend(title='SVM')


# The combination of the feature values of the points touching the margins define the 
# ***support vectors*** because they completely determine both the decision line and the width of the gap, i.e. they **support** the decision line. 
# In the context of SVMs, the perpendicular distance between any support vector and the decision line is called the ***margin*** 
# (so the width of the gap is twice the margin).


# At this point it becomes clear how the SVM works: The algorithm searches for the **decision line with the largest perpendicular distance to the *support vectors* (*margin*)**.


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

features =scaler.fit_transform(df.loc[:, ['feature_1', 'feature_2']])
target = df.loc[:, 'class']
model_svm.fit(features, target)



print(model_svm.support_vectors_)
print(model_svm.support_)
features[model_svm.support_, :]

# Equals, needed to check
model_svm.support_vectors_ = features[model_svm.support_, :]


# Decision license
model_svm.coef_



# SVM flexibility: hard margin and soft margin

new_row = {"feature_1": 4.1, "feature_2": -5.7, 'class': 0}
df.append(new_row, ignore_index=True)

features = scaler.fit_transform(df.loc[:, ['feature_1', 'feature_2']])  # fit scaler to the new data and transform the data


# Remember: Small C values result in more points in the "gap" (margin) and high C values result in fewer points.


# C Parameter
# Allow the SVM to look for the largest margin to the second closest (or third, fourth, fifth, etc.) support vectors
# We say the SVM has a hard margin when it doesn't let any points into the "gap", and we call it a soft margin when they are allowed. 
# Sometimes we say that C controls the penalty level for incorrect classifications.



model_svm = SVC(kernel='linear', C=1000)
model_svm.fit(features, df.loc[:, 'class'])


# You can find the best C parameter for your own data with a grid search

# Creating usable features from it part of what's called Natural Language Processing (NLP for short). We'll take a closer look at how this works in the next lesson.


# Remember:

# Support vector machines are ideal for small or medium data sets.
# The support vector machine creates a decision plane that is supported by the support vectors.
# A high value for C results in a hard margin, and a low value in a soft margin.