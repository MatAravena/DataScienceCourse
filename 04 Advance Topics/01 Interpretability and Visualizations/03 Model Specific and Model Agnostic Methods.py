# Model Specific and Model Agnostic Methods


# You will know the difference between model-specific and model-agnostic methods.
# You will know what permutation feature importance means.
# You will be able to calculate permutation feature importance.


# # Global interpretation methods
# Model-specific methods: Functions built into a model that calculate feature importance.
# Model-agnostic methods: General strategies for obtaining feature importance. Can be applied to any model.


# # Model-specific methods
# In the field of machine learning, a **black box model** is a model that cannot be interpreted locally.
# Black box models often provide significantly better predictions, but it's difficult to say how the algorithm arrived at these results.
# This in turn makes effective data storytelling more difficult.

# Random forests are an example of a black box model.

# To interpreted the black box there are ways like the atributte  `.feature_importances_` in trees algorithms
# this is to get the feature importance of the decision tree.
# So it is a model-specific method for tree algorithms.


# read data
import pandas as pd
df_train = pd.read_csv('attrition_train.csv')

# split training data into features and target
features_train = df_train.drop('attrition',axis=1)
target_train = df_train.loc[:, 'attrition']

df_train.head()

from sklearn.ensemble import RandomForestClassifier
model_rf = RandomForestClassifier(class_weight='balanced', random_state=0, max_depth=12, n_estimators= 100)
model_rf.fit(features_train,target_train)


# **Deep Dive:** Now you can briefly consider what would happen if you interpreted this kind of model locally. 
# For the decision tree we simply followed the decision rules and arrived at a prediction after reaching the maximum depth of 3. 
# Here we would have to do this for 100 decision trees with a depth of 12. It really would be too confusing and we don't recommend trying it. 
# Only global methods have a chance here.

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

fig, ax = plt.subplots(figsize=(8, 6))

# Convert feature importance array into a series and visualize
feature_importance = pd.Series(model_rf.feature_importances_,
                               index=features_train.columns
                              ).sort_values()
feature_importance.plot(kind='barh', ax=ax)
fig.tight_layout()

# It's important to understand here that each model can bring a new perspective so you should always look at several models. 
# This makes it even more important to be able to interpret each model so that you can create a consistent and convincing data story.



# Feature importances only make sense in the context prediction quality they are associated with. 
# You shouldn't create a data story based on a model with bad predictions. Let us look at the quality of our random forest.
# read data
import pandas as pd
df_test = pd.read_csv('attrition_test.csv')

# split training data into features and target
features_test = df_test.iloc[:, 1:]
target_test = df_test.loc[:, 'attrition']

from sklearn.metrics import precision_score, recall_score
target_test_pred = model_rf.predict(features_test)

print('Precision: ', precision_score(target_test, target_test_pred))
print('Recall: ', recall_score(target_test, target_test_pred))




# Permutation feature importance
# Examples support vector machine with a polynomial kernel
# there is not methods attributes like in decision tree that's why the aprouch is to get the importances of the features with am agnostic methods.
# one of those methods is permutation feature importance.


# The order is also important here: 
# We want to find out how much a single feature contributes to our predictions by training two models. 
# One is trained with all the features as usual and provides the "real" results. 
# The other model is also trained with all the features, whereby the connection between a single feature and the target is destroyed. 
# Then you compare the predictions. The greater the deviations, the more important the feature is.



# Technically, the following steps are performed:

# 1 The model, including hyperparameters, is fitted to the data as usual.
# 2 The prediction quality  𝑉𝑜𝑟𝑖𝑔
#   is calculated with whichever metric you select.
# 3 The values of a single feature, i.e. a single column, are mixed randomly, so the link between this feature and the target is lost. 
#   No values are deleted or generated, they are only shuffled, so to speak.
# 4 The new prediction quality  𝑉𝑝𝑒𝑟𝑚
#   is calculated based on the new data, i.e. with the permuted (mixed) column.
# 5 The feature importance is calculated either as the ratio  𝑉𝑜𝑟𝑖𝑔 / 𝑉𝑝𝑒𝑟𝑚,
#   or as the difference  𝑉𝑜𝑟𝑖𝑔−𝑉𝑝𝑒𝑟𝑚. 6 Steps 2-5 are repeated for each feature and the final result is presented as a data story.




# Try with the metric accuaricy    -->   𝑉 𝑜𝑟𝑖𝑔𝑖𝑛𝑎𝑙
from sklearn.metrics import accuracy_score
accuracy_score(target_test, target_test_pred)

# try to create an new randomized dataset to test with new values from the previous test dataframe 
features_test_perm = features_test.copy()

# Shufle values within a column
age_series_perm = features_test_perm.loc[:, 'age'].sample(frac=1, replace=False, random_state=0)
age_series_perm.head()

# reorganized the previous randomized index in the serie
age_series_perm = age_series_perm.reset_index(drop=True) 

# Get the new predictions
features_test_perm.loc[:, 'age'] = age_series_perm
target_test__pred_perm = model_rf.predict(features_test_perm)

acc_age =  accuracy_score(target_test, target_test__pred_perm)
print('accuracy with age permuted', acc_age)

# Compare the metric
acc_orig - acc_age
# 0.01133786848072571



# Permute in all columns
perm_importances = []
features_test_perm = features_test.copy()

for x in features_test_perm.columns:
    #permutate feature column
    newCol_Shufle = features_test_perm.loc[:,x].sample(frac=1, replace=False, random_state=0)
    features_test_perm.loc[:, x] = newCol_Shufle.reset_index(drop=True)
    
    target_test__pred_perm = model_rf.predict(features_test_perm)

    acc_perm =  accuracy_score(target_test, target_test__pred_perm)

    perm_importances.append(acc_orig-acc_perm)

    # reset permutation
    features_test_perm.loc[:, x] = features_test.loc[:, x]

perm_importances


# Grpah
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlabel('Permutation Feature Importance')

# Convert feature importance array into a series and visualize
perm_importance = pd.Series(perm_importances, index=features_test.columns).sort_values()
perm_importance.plot(kind='barh', ax=ax)

fig.tight_layout();


# The values may differ for other permutations. For increased accuracy, you can repeat this process several times and use calculate the average for the results.

# You can use the permutation feature importance with any metric or model to gain insight into the model, 
# because the method only works with the end results and does not take model-internal properties into account. Truly model agnostic!


# **Note:** While the permutation feature importance may seem like a universal technique for global interpretation, it has important drawbacks.
# 1. Assume that two features are strongly correlated. If you now shuffle one of the two, you create data that could never exist. The permuted prediction would therefore 
#     never be relevant and would be much lower than for uncorrelated features. So the feature importance would be valued too highly. 
#     you should always keep this in mind when making recommendations.

# 2. It's unclear whether the permutation feature importance should be determined from the training or test data. In our case, we determined 
#     it from the test data, but you could criticize the fact that the model was always trained on unpermuted data, and so it's unclear how 
#     realistic the feature importances are. So you should test several possibilities (also shuffle the training data) to get a better overview.
#     The permutation feature importance is thus always only a statement about the tendency.




# Remember:
# There are model-specific and model agnostic methods for global interpretability.
# You can use permutation feature importance to estimate the importance of the features
# Shuffling the values in a column destroys the direct link between the column and the target variable

