# Grid search with k-Nearest Neighbors, logistic regression and random forest
# Grid search with an ensemble classifier



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


# Deleting warnings
from sklearn.exceptions import DataConversionWarning, UndefinedMetricWarning
import warnings
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)
warnings.filterwarnings(action='ignore', category=DeprecationWarning)


# You've got to know three different classification approaches so far:
# k-Nearest Neighbors.
# Logistic Regression.
# Decision trees and forests.

# k-Nearest Neighbors
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.pipeline import Pipeline

pipeline_knn= Pipeline([('std', StandardScaler ()),('knn', KNeighborsClassifier ())])

import numpy as np
k = np.unique(np.geomspace(1, 500, 15, dtype='int'))  # create 15 values between 1 and 500 with increasing distance

search_space_knn = {'knn__n_neighbors': k,  # use the created values as number of neighbors
                    'knn__weights': ['uniform', 'distance']}


from sklearn.model_selection import GridSearchCV

model_knn = GridSearchCV(estimator=pipeline_knn, 
                         param_grid=search_space_knn,
                         scoring='f1',
                         cv=5)
model_knn.fit(features_train,target_train)
print(model_knn.best_estimator_)
print(model_knn.best_score_)

# Pipeline(steps=[('std', StandardScaler()),
#                 ('knn', KNeighborsClassifier(n_neighbors=1))])
# 0.28101947510672154

# The best model only uses one neighbor (n_neighbors=1). The parameter weights is not shown since it is set to the default value 'uniform'.



# Logistic Regression
from sklearn.linear_model import LogisticRegression
pipeline_log = Pipeline([('scaler', StandardScaler()),
                         ('log', LogisticRegression(solver='saga',
                                                    max_iter=1e4,
                                                    class_weight='balanced', 
                                                    random_state=42))])

C_values = np.geomspace(start=0.001, stop=1000, num=14)

search_space_log = {'log__penalty': ['l1', 'l2'],
                    'log__C': C_values}

model_log = GridSearchCV(estimator=pipeline_log,
                         param_grid=search_space_log,
                         scoring='f1',
                         cv=5)

model_log.fit(features_train, target_train)

print(model_log.best_estimator_)
print(model_log.best_score_)

# Pipeline(steps=[('scaler', StandardScaler()),
#                 ('log',
#                  LogisticRegression(C=0.2030917620904737,
#                                     class_weight='balanced', max_iter=10000.0,
#                                     penalty='l1', random_state=42,
#                                     solver='saga'))])
# 0.4730589275938115

# The best model uses quite a lot of regularization (C=0.20) like in a LASSO regression (penalty='l1') and results in an F1 score of 47.3%.




# The random forest

from sklearn.ensemble import RandomForestClassifier
search_space_rf = {'max_depth': np.geomspace(start=3, stop=50, num=10, dtype='int'),
                   'min_samples_leaf': np.geomspace(start=1, stop=500, num=10, dtype='int')}

model_rf = GridSearchCV(estimator=RandomForestClassifier(class_weight='balanced',
                                                         n_estimators=50,
                                                         random_state=42),
                        param_grid=search_space_rf,
                        scoring='f1',
                        cv=5)

model_rf.fit(features_train, target_train)

print(model_rf.best_estimator_)
print(model_rf.best_score_)

# RandomForestClassifier(class_weight='balanced', max_depth=7, min_samples_leaf=7,
#                        n_estimators=50, random_state=42)
# 0.49027633482249566

# The best model uses seven decision levels (max_depth=7) with at least seven data points in each leaf (min_samples_leaf=7). 
# The resulting model quality of approx. 49% according to the F1 score is the best value so far. 



# Combining classification models optimally

# VotingClassifier 
from sklearn.ensemble import VotingClassifier

# voting parameter controls whether to combine the predicted categories (voting='hard') or the predicted category probabilities (voting='soft')
# weights parameter to specify whether all the models should contribute equally to the final vote (weights=None) or whether they should be weighted (weights=[weight_model_1, weight_model_2, weight_model_3])

search_space_ens = {'voting':['soft','hard'],
                    'weights':[None, [model_knn.best_score_, model_log.best_score_, model_rf.best_score_]]}

voting_knn_log_rf = VotingClassifier(estimators=[('knn', model_knn), ('log', model_log), ('rf', model_rf)])

model_ens = GridSearchCV(estimator=voting_knn_log_rf,
                         param_grid=search_space_ens,
                         scoring='f1',
                         cv=3,
                         n_jobs=-1)

model_ens.fit(features_train, target_train)

print(model_ens.best_estimator_)
print(model_ens.best_score_)
print(model_ens.best_params_)

# VotingClassifier(estimators=[('knn',
#                               GridSearchCV(cv=5,
#                                            estimator=Pipeline(steps=[('std',
#                                                                       StandardScaler()),
#                                                                      ('knn',
#                                                                       KNeighborsClassifier())]),
#                                            param_grid={'knn__n_neighbors': array([  1,   2,   3,   5,   9,  14,  22,  34,  54,  84, 132, 205, 320, 499]),
#                                                        'knn__weights': ['uniform',
#                                                                         'distance']},
#                                            scoring='f1')),
#                              ('log',
#                               GridSearchCV(cv=5,
#                                            estimator=Pipeline(steps=[('scaler',
#                                                                       StandardScaler()...
#                                                           4.92388263e+00, 1.42510267e+01, 4.12462638e+01, 1.19377664e+02,
#                                                           3.45510729e+02, 1.00000000e+03]),
#                                            'log__penalty': ['l1',
#                                                                         'l2']},
#                                            scoring='f1')),
#                              ('rf',
#                               GridSearchCV(cv=5,
#                                            estimator=RandomForestClassifier(class_weight='balanced',
#                                                                             n_estimators=50,
#                                                                             random_state=42),
#                                            param_grid={'max_depth': array([ 3,  4,  5,  7, 10, 14, 19, 26, 36, 49]),
#                                                        'min_samples_leaf': array([  1,   1,   3,   7,  15,  31,  62, 125, 250, 499])},
#                                            scoring='f1'))])
# 0.49335232668566004
# {'voting': 'hard', 'weights': None}
# Hint
# Run code: Ctrl + Enter
# Reset
# The best meta-model uses the categorical predictions (voting='hard') and weights all models equally (weights=None). The resulting model quality of 49.3% according to the F1 score is about as high as with logistic regression and the random forest.



# Evaluating the best classification models
from sklearn.metrics import precision_score, recall_score, f1_score

for clf in [model_log, model_rf, model_ens]:
    
    target_test_pred = clf.predict(features_test)
    
    print('\nPrecision: ', precision_score(target_test, target_test_pred))
    print('Recall: ', recall_score(target_test, target_test_pred))
    print('F1: ', f1_score(target_test, target_test_pred))



# The final data pipeline
# merge all the steps into a final data pipeline



# Remember:

# Ensembling with sklearn.ensemble.VotingClassifier
# VotingClassifier expects a list of tuple pairs of name and models for estimators.
# Meta models can also be optimized with a grid search.
