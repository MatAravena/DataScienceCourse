# Ensembling
# Bagging
# Random Forests

# Roughly speaking, there are three approaches to combine machine learning models


# 1. Ensembling: Combining predictions so that the meta-model provides more stable predictions (implemented in sklearn with from sklearn.ensemble.VotingClassifier)

# 2. Stacking: Using predictions from some models as features for a higher-level model to generate better predictions (implemented in the vecstack module)

# 3. Boosting: A series of models is trained so that that the weight of data points, which were falsely classified in the previous step of the series, is heavier. 
# So the boosting model as a whole focuses on the difficult data points, in the hope that this will improve the overall predictions (implemented in sklearn with sklearn.ensemble.AdaBoostClassifier, for example)

#  We know that they have totally different approaches to generating predictions (neighbourhood voting, regression to log odds, decision rules). Instead, in this exercise we'll focus on how the very popular random forest algorithm is an ensemble of decision trees.


# random forest uses two strategies to give chance more influence on the decision trees:

# -->Bootstrap aggregating
# -->Random feature selection


# Preparing data training
import pandas as pd
import pickle

#load pipeline
pipeline = pickle.load(open("pipeline.p",'rb'))
col_names = pickle.load(open("col_names.p",'rb'))

#gather data
df_train = pd.read_csv('attrition_train.csv')
df_test = pd.read_csv('attrition_test.csv')

#extract features and target
features_train = df_train.drop('attrition', axis=1)
features_test = df_test.drop('attrition', axis=1)

target_train = df_train.loc[:,'attrition']
target_test = df_test.loc[:,'attrition']

#transform data
features_train = pd.DataFrame(pipeline.transform(features_train), columns=col_names)
features_test = pd.DataFrame(pipeline.transform(features_test), columns=col_names)

# look at raw data
features_train.head()

#combine features and target
df_train = pd.concat([features_train, target_train], axis=1)
df_train.head()

list_df_train_bagging = []

for i in range(100):
    list_df_train_bagging.append(df_train.sample(frac=1, replace=True, random_state=i))

list_features_train_bagging = []
list_target_train_bagging = []

for df_tmp in list_df_train_bagging:
    list_features_train_bagging.append(df_tmp.drop('attrition', axis=1))
    list_target_train_bagging.append(df_tmp.loc[:, 'attrition'])



# A simple version of a random forest

# Random feature selection: Each time you want a decision tree to generate a decision rule, it only has a few randomly selected features available to it

from sklearn.tree import DecisionTreeClassifier
tree_models = []
for x in range(len(list_df_train_bagging)): 
    model_tmp = DecisionTreeClassifier(max_depth=4, class_weight='balanced',splitter='random',random_state=i)
    model_tmp.fit(list_features_train_bagging[i], list_target_train_bagging[i])
    tree_models.append(model_tmp)


predictions= {}  # create dict to hold the predictions of each decision tree
for i, model_tmp in enumerate(tree_models):  # use enumerate to get numbers from 0 to 99
    # save the predictions of each tree for category attrition = 1 via [:, 1]
    predictions['model_{}'.format(i)] = model_tmp.predict_proba(features_test)[:, 1] 

df_target_test_pred_proba = pd.DataFrame(predictions)  # convert the dict to a df

pd.set_option('precision', 2)  # show only 2 decimals

df_target_test_pred_proba.head()



from sklearn.metrics import precision_score, recall_score

print('Precision: ', precision_score(target_test, df_target_test_pred_proba.mean(axis=1).round()))
print('\nRecall: ', recall_score(target_test, df_target_test_pred_proba.mean(axis=1).round()))

# Model	                        precision	    recall
# df_target_test_pred_proba	    approx. 46%	    approx. 53%



# The random forest
from sklearn.ensemble import RandomForestClassifier
model_rf = RandomForestClassifier(n_estimators=100, max_depth=4, class_weight='balanced', random_state=42)

model_rf.fit(features_train,target_train)

predictRandomForest = model_rf.predict(features_test)

print('Precision: ', precision_score(target_test, predictRandomForest))
print('Recall: ', recall_score(target_test, predictRandomForest))

# Model	                    precision	    recall
# df_target_test_pred_proba	approx. 46%	    approx. 53%
# model_rf	                approx. 45%	    approx. 50%




# Remember:

# from sklearn.ensemble import RandomForestClassifier
# The decision trees in a random forest differ from normal decision trees in two ways: training data bagging and the feature selection when generating a decision line
# The more decision trees in the random forest, the better (n_estimators parameter of RandomForestClassifier)