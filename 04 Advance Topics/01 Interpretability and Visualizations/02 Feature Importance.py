# How you can calculate the importance of features in decision trees.
# How you can represent the importance of features.
# How data stories work


# # Determining feature importance in the decision tree


import pandas as pd
df_train = pd.read_csv('attrition_train.csv')
df_train.head()

# split training data into features and target
features_train = df_train.drop('attrition',axis=1)
target_train = df_train.loc[:, 'attrition']

# instantiate and fit a decision tree
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(class_weight='balanced', max_depth=3, random_state=0, min_samples_leaf=20)
model.fit(features_train, target_train)

# **Gini importance**
# The higher the *Gini importance*, the more important the feature is; *Gini impurity* and *Gini importance* are related, 
# but have different meanings. The former can be interpreted as the probability of misclassification of a group of data points, 
# while the latter measures the relevance of a feature, if this probability is reduced.

# print
model.feature_importances_

# How to visualize the most important features
feature_importance = pd.Series(model.feature_importances_, index=features_train.columns)
feature_importance.sort_values(ascending=False)





# # Displaying feature importance in the decision tree with matplotlib

import matplotlib.pyplot as plt

colors=['#99cced']*5 + ['#17415f']
colors = ['#99cced']*5 + ['#17415f']

mask = feature_importance > 0
feature_importance = feature_importance.loc[mask].sort_values()

fig, ax = plt.subplots(figsize=(10, 4))

feature_importance.plot(kind='barh', color=colors, width=0.8);


# Giving texts to the axis
ax.set_title(label='Overworked employees are more willing to change employers',
             family='serif',
             color=colors[-1],
             weight='semibold',
             size=14)
ax.set_xlabel('Relative importance',
              size=12, position=[0, -2],
              horizontalalignment='left')
ax.set_ylabel('Features',
              size=12, position=[0, 1],
              horizontalalignment='right')

# Incrising the text in the Y axis
ax.set_yticklabels(ax.get_yticklabels(), size=12)

#remove top and right frame parts
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)



# Adding labels next to each bar
for idx in range(len(feature_importance.index)):
    ax.text(s='{}%'.format(int(100*feature_importance.iloc[idx])),
            x=feature_importance.iloc[idx]+0.005,
            y=idx, 
            size=12,
            color='black')

#remove x-ticks and spine
ax.spines['bottom'].set_visible(False)
ax.set_xticks([])
ax.set_xticklabels([])

fig


# Remember:

# The Gini importance measures the weighted contribution a feature makes in reducing false predictions.
# You can get them from the decision tree my_model.feature_importance_.
# Make your images as user-friendly as possible.