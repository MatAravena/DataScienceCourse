# How to instantiate a decision tree model
# How a decision tree ends up with its category predictions
# How to simplify a decision tree


# Your first decision tree
import pickle
#load pipeline
pipeline = pickle.load(open("pipeline.p",'rb')) 
col_names = pickle.load(open("col_names.p",'rb'))

import pandas as pd

#gather data
df_train = pd.read_csv('attrition_train.csv')
df_test = pd.read_csv('attrition_test.csv')

#extract features and target
features_train = df_train.drop('attrition', axis=1)
features_test = df_test.drop('attrition', axis=1)

target_train = df_train.loc[:,'attrition']
target_test = df_test.loc[:,'attrition']

#transform data
features_train = pd.DataFrame(pipeline.transform(features_train), columns=col_names) # already fitted
features_test = pd.DataFrame(pipeline.transform(features_test), columns=col_names)

# look at raw data
features_train.head()


# DecisionTreeClassifier(criterion=str, # impurity measurement to minimize at each split
#                        max_depth=int,         # restricts how consecutive splits are made in the tree
#                        min_samples_split=int, # minimum observations that need to be left in a node to consider splitting
#                        min_samples_leaf=int,  # minimum observations that need to be in the terminal nodes (leafs)
#                        max_features=int,      # maximum number of features to consider at each split
#                        random_state=int,      # set for reproducibility
#                        max_leaf_nodes=int)    # max number of terminal nodes (leaves)

from sklearn.tree import DecisionTreeClassifier
model_1 = DecisionTreeClassifier(max_depth=1, random_state=0)


num_cols = ['age', 
            'distancefromhome', 
            'monthlyincome', 
            'numcompaniesworked', 
            'percentsalaryhike', 
            'trainingtimeslastyear',
            'pca_years_0',
            'pca_years_1']

features_train = features_train.loc[:, num_cols]


# Tip: The data does not have to be standardized beforehand for a decision tree.
model_1.fit(features_train, target_train)

# plot_tree(decision_tree=var, #decision tree model to be plotted
#           feature_names=list, #names of each of the features
#           class_names=list, #names of the target classes [class0 , class1, ...]
#           filled=bool) #coloration of majority class for classification

from sklearn.tree import plot_tree
plot_tree(decision_tree=model_1 , 
        feature_names=features_train.columns,
        filled=True);
import matplotlib.pyplot as plt

# histogramm of first principal component
fig, ax = plt.subplots(figsize=(8,6))
features_train.loc[:, 'pca_years_0'].plot(kind='hist',
                                          bins=100,
                                          ax=ax)

# grey vertical line
ax.axvline(-2.219, # tree uses -2.219 as decision line
           ls='--', 
           color = "grey")

# set title and label axes
ax.set(xlabel='Feature value (\'pca_years_0\')',
       ylabel='Count',
       title='Decision tree with max_depth=1');


# The criterion parameter controls which criterion is used to achieve the purest possible leaves. 
# You can choose between 'gini' and 'entropy'. These two criteria give the same results in 98% of cases, 
# but 'entropy' can use more computing resources, which is why the default 'criterion='gini' is often used



# the Gini impurity is a measure of the impurity of a class distribution



# Interpret threshold values
#get column names from PCA
pca_cols = pipeline._columns[0]

# mask in piped training set
mask = features_train['pca_years_0']<=-2.219

# import relevant columns from original untransformed training data
df_train_original = pd.read_csv('attrition_train.csv', usecols=pca_cols)

#group and aggregate
df_train_original.groupby(mask).median().T




import seaborn as sns

#select columns
cols = ['years_currentrole','totalworkingyears']

fig, axs = plt.subplots(len(cols),
                        figsize=((6,10)),
                        sharey=False)
#plot
for i, col in enumerate(cols):
    sns.histplot(df_train_original[mask].loc[:,col],
                 stat="density", # use relative frequncies
                 kde=False,      # mute kde-plot
                 label='leave',  # set label for legend
                 ax=axs[i],      # use subplot
                 color="#3399db")
    sns.histplot(df_train_original[~mask].loc[:,col], 
                 stat="density",
                 kde=False, 
                 label='stay',
                 ax=axs[i],
                 color="#ff9e1c")
    axs[i].set(ylabel='Rel. frequency [%]')
    axs[i].legend(title="Prediction") #title for legend

#prettify plot
fig.suptitle("Feature Distributions") #title for whole figure
fig.subplots_adjust(top=0.92) #set main title position




# Decision trees with several decision levels

model_2 = DecisionTreeClassifier(max_depth=2, random_state=0)
model_2.fit(features_train, target_train)

fig, ax = plt.subplots(figsize=(8,6))
sns.scatterplot(data=features_train, 
                x='pca_years_0', 
                y='age', 
                hue=target_train,
                alpha=0.3, 
                ax=ax, 
                palette=['#ff9e1c', '#3399db'])

# grey vertical line
ax.axvline(-2.219, 
           ls='--', 
           linewidth=3,
           color = "grey")

# grey horizontal line
ax.plot([-3,-2.219],
        [31.5, 31.5], 
        ls='--', 
        linewidth=3,
        color = "grey")

# set title and labels
ax.set(xlabel='Feature value (\'pca_years_0\')',
       ylabel='Employee age',
       title='Decision tree with max_depth=2');


# By default, the decision tree generates as many decision rules as it needs to obtain leaves that are pure. 
# Pure leaves only contain data points in one category. Let's also generate a maximum-depth tree so that we can compare as many different kinds of trees.

model_max = DecisionTreeClassifier(random_state=0)
model_max.fit(features_train, target_train)



# Evaluating decision trees

features_test = features_test.loc[:, num_cols]

target_test_pred_1 = model_1.predict(features_test)
target_test_pred_2 = model_2.predict(features_test)
target_test_pred_max = model_max.predict(features_test)

from sklearn.metrics import precision_score, recall_score

print('Precision:')
print('1 decision level: ', precision_score(target_test, target_test_pred_1))
print('2 decision levels: ', precision_score(target_test, target_test_pred_2))
print('max decision levels: ', precision_score(target_test, target_test_pred_max))

print('\nRecall:')
print('1 decision level: ', recall_score(target_test, target_test_pred_1))
print('2 decision levels: ', recall_score(target_test, target_test_pred_2))
print('max decision levels: ', recall_score(target_test, target_test_pred_max))


# Precision:
# 1 decision level:  0.5172413793103449
# 2 decision levels:  0.5416666666666666
# max decision levels:  0.23684210526315788

# Recall:
# 1 decision level:  0.21428571428571427
# 2 decision levels:  0.18571428571428572
# max decision levels:  0.2571428571428571

# The decision tree size increases, the precision score measured from the test data decreases, while the recall score increases slightly. 
# The difference in the precision indicates that the decision tree could suffer from overfitting as the decision levels increase. 
# To verify this, it makes sense to look at the validation curves, 



# Remember:

# Import decision tree model: from sklearn.tree import DecisionTreeClassifier
# Instantiate model with x decision levels: model = DecisionTreeClassifier(max_depth=x)
# A maximum-depth tree can suffer from overfitting.