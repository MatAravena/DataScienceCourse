# The False Positive Rate model quality metric
# The Receiver Operating Characteristics curve
# The Area under the ROC Curve model quality Metric (ROC AUC)



# Model quality metrics for binary classification
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
df_train = onehot.fit_transform(df_train) # fit and transform to training set

# Look at data
df_train.head()


# model_log without regularization
# model_reg with regularization

# 1. Model specification
from sklearn.linear_model import LogisticRegression

###################################################################
# a) Without regularization

# 2a. Feature matrix and target vector
features_train = df_train.iloc[:, 1:]
target_train = df_train.loc[:, 'fake']

# 3a. Model instantiation
model_log = LogisticRegression(solver='lbfgs', max_iter=1e4, C=1e42, random_state=42)

# 4a. Model fitting
model_log.fit(features_train, target_train)

#####################################################################
# b) With regularization

# 2b. Feature matrix standardization 
from sklearn.preprocessing import StandardScaler #use StandardScaler to adjust the features

scaler = StandardScaler()
features_train_scaled = scaler.fit_transform(features_train) #fit to training data and scale it

# 3b. Model instantiation
model_reg = LogisticRegression(solver='lbfgs', max_iter=1e4, C=0.5, random_state=42)

# 4b. Model fitting
model_reg.fit(features_train_scaled, target_train)




# 4 model quality metrics for classification algorithms:

# Accuracy: sklearn.metrics.accuracy_score()
# Precision: sklearn.metrics.precision_score()
# Recall: sklearn.metrics.recall_score()
# F1 score: sklearn.metrics.f1_score()


# To avoid overfitting, you shouldn't use the training data to evaluate the model.

# Prepare test data
# import data
df_test = pd.read_csv("social_media_test.csv", index_col=[0])

# label encoding
dict_label_encoding = {'Yes': 1, 'No': 0}
df_test.loc[:, 'profile_pic'] = df_test.loc[:, 'profile_pic'].replace(dict_label_encoding)
df_test.loc[:, 'extern_url'] = df_test.loc[:, 'extern_url'].replace(dict_label_encoding)
df_test.loc[:, 'private'] = df_test.loc[:, 'private'].replace(dict_label_encoding)

# one-hot encoding
df_test = onehot.transform(df_test) #transform to test set

# look at data
df_test.head()



# Test models

# Logaritmic Reggresion
# features matrix and target vector
features_test = df_test.iloc[:, 1:]
target_test = df_test.loc[:, 'fake']

# predict target values from model
target_test_pred_log = model_log.predict(features_test)

# model evaluation
from sklearn.metrics import precision_score, recall_score
precision_log = precision_score(target_test, target_test_pred_log)
recall_log = recall_score(target_test, target_test_pred_log)

# print
print('Precision of model without regularisation: ', precision_log)
print('Recall of model without regularisation: ', recall_log)
# Precision of model without regularisation:  0.8666666666666667
# Recall of model without regularisation:  0.8666666666666667

# Regression
# features matrix and target vector
features_test = df_test.iloc[:, 1:]
target_test = df_test.loc[:, 'fake']

features_standarized_test = scaler.transform(features_test)

# predict target values from model
target_test_pred_reg = model_reg.predict(features_standarized_test)

# model evaluation
from sklearn.metrics import precision_score, recall_score
precision_reg = precision_score(target_test, target_test_pred_reg)
recall_reg = recall_score(target_test, target_test_pred_reg)

# print
print('Precision of model without regularisation: ', precision_reg)
print('Recall of model without regularisation: ', recall_reg)
# Precision of model without regularisation:  0.8813559322033898
# Recall of model without regularisation:  0.8666666666666667

# Model	precision	recall
# model_log	86.7%	86.7%
# model_reg	88.1%	86.7%

# model_log is equally good at identifying fake accounts as such (recall), while model_reg had more accounts that were actually 
# fake among the accounts that were predicted to be fake (precision). In other words, the predictions from model_log are slightly 
# less reliable than those from model_reg.




# ROC Curve
# describes how the recall and false positive rate change with the threshold value.


# module import
from sklearn.metrics import confusion_matrix

# calculate confucion matrices
cm_log = confusion_matrix(target_test, target_test_pred_log)
cm_reg = confusion_matrix(target_test, target_test_pred_reg)

print('False positive rate of model without regularisation: ', 
      round(cm_log[0, 1]/(cm_log[0, 1] + cm_log[0, 0]), 3))

print('False positive rate of model with regularisation: ', 
      round(cm_reg[0, 1]/(cm_reg[0, 1] + cm_reg[0, 0]), 3))




# calculate probability
target_test_pred_proba_log = model_log.predict_proba(features_test)

# module import
from sklearn.metrics import roc_curve

# calculate roc curve values
false_positive_rate_log, recall_log, threshold_log = roc_curve(target_test, 
                                                               target_test_pred_proba_log[:, 1],
                                                               drop_intermediate=False) 

# roc_curve() returns three vectors as arrays:

# the false positive rate (FPR) for each threshold
# the recall for each threshold
# the thresholds


print("Threshold (model_log): ", threshold_log[49])
print("Recall (model_log): ", recall_log[49])
print("False positive rate (model_log): ", false_positive_rate_log[49])


print("Threshold (model_log): ", threshold_log[70])
print("Recall (model_log): ", recall_log[70])
print("False positive rate (model_log): ", false_positive_rate_log[70])




# module import and style setting
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

# figure and axes intialisation
fig, ax = plt.subplots()

# reference lines
ax.plot([0, 1], ls = "--", label='random model')  # blue diagonal
ax.plot([0, 0], [1, 0], c=".7", ls='--', label='ideal model')  # grey vertical
ax.plot([1, 1], c=".7", ls='--')  # grey horizontal

# roc curve
ax.plot(false_positive_rate_log, recall_log, label='model_log')

# labels
ax.set_title("Receiver Operating Characteristic")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("Recall")
ax.legend()






# calculate probability
target_test_pred_proba_reg = model_reg.predict_proba(features_test_scaled)

# calculate roc curve values
false_positive_rate_reg, recall_reg, threshold_reg = roc_curve(target_test,
                                                               target_test_pred_proba_reg[:, 1],
                                                               drop_intermediate=False) 

# figure and axes intialisation
fig, ax = plt.subplots()

# reference lines
ax.plot([0, 1], ls = "--", label='random model')  # blue diagonal
ax.plot([0, 0], [1, 0], c=".7", ls='--', label='ideal model')  # grey vertical
ax.plot([1, 1], c=".7", ls='--')  # grey horizontal

# roc curve
ax.plot(false_positive_rate_reg, recall_reg, label='model_reg')

# labels
ax.set_title("Receiver Operating Characteristic")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("Recall")
ax.legend()


# ROC AUC metric

# ROC AUC
# receiver operator characteristic area under the curve

from sklearn.metrics import roc_auc_score

roc_auc_score(target_test, target_test_pred_proba_log[:, 1])
# 0.9624999999999999

roc_auc_score(target_test, target_test_pred_proba_reg[:,1])
# 0.9322222222222222

# The value is slightly lower than for model_log, at around 93%.
# This again confirms our visual impression above: The model with regularization follows the ideal less than the model without regularization.

# So in this case, regularization has reduced rather than improved the quality of the model. Regardless of the decision thresholds we choose,
# we can conclude that we have overfitted our model!



# Remember:

# The confusion matrix (sklearn.metrics.confusion_matrix()) and all model quality metrics derived from it are based on predicted categories
# You can calculate false positive and false negative rates from the confusion matrix
# The ROC curve and the ROC AUC metric are based on the predicted probabilities
# sklearn function of the model quality metric ROC AUC: sklearn.metrics.roc_auc_score()
