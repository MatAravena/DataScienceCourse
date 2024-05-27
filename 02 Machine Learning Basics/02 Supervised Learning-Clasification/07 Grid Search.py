# module import
import pandas as pd

# data gathering
df_train = pd.read_csv('occupancy_training.txt')

# turn date into DateTime
df_train.loc[:, 'date'] = pd.to_datetime(df_train.loc[:, 'date'])

# turn Occupancy into category
df_train.loc[:, 'Occupancy'] = df_train.loc[:, 'Occupancy'].astype('category')

# define new feature
df_train.loc[:, 'msm'] = (df_train.loc[:, 'date'].dt.hour * 60) + df_train.loc[:, 'date'].dt.minute

# take a look
df_train.head()



# to convert int numbers to float numbers, and will give you a warning each time that you can safely ignore
# module import
from sklearn.exceptions import DataConversionWarning
import warnings

warnings.filterwarnings('ignore', category=DataConversionWarning)  # suppress data conversion warnings



from sklearn.pipeline import  Pipeline
from sklearn.preprocessing import  StandardScaler
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.model_selection import  cross_val_score

features_train = df_train.loc[:, ['CO2', 'msm']]
target_train = df_train.loc[:, 'Occupancy']

pipeline_std_knn = Pipeline([('std', StandardScaler()),
                             ('knn', KNeighborsClassifier(n_neighbors=1))])

cv_results = cross_val_score(estimator=pipeline_std_knn,#pipeline or Model
                            X=features_train,           #feature matrix
                            y=target_train,             #target values
                            cv=2,                       #number of folds for cross-validation
                            scoring='f1',               #scoring function e.g ['accuracy','f1','recall']
                            n_jobs=-1)                  #number of cpu cores to use for computation (use -1 to use all available cores)
cv_results.mean()


import numpy as np
k = np.geomspace(1, 1000, 15, dtype='int')
k = np.unique(k)



# Finding optimal hyperparameters with a grid search
from sklearn.model_selection import GridSearchCV
search_space = {'knn__n_neighbors': k}

grid_search_for_k = GridSearchCV(estimator=pipeline_std_knn,  # estimator
             param_grid=search_space,                         # the grid of the grid search
             scoring='f1',                                    # which measure to optimise on
             cv=2,                                            # number of folds during cross-validation
             n_jobs=-1)                                       # number of CPU cores to use (use -1 for all cores)

# Then we can fit this variable to the data. In our grid search, this corresponds to finding the best k value for k-nearest neighbors
grid_search_for_k.fit(features_train, target_train)

# all results specified by training a model are written with an underscore at the end of an attribute.
grid_search_for_k.best_score_
grid_search_for_k.best_estimator_


# Finding optimal hyperparameters with a grid search
search_space_grid = {'knn__n_neighbors' :  k ,
                      'knn__weights': ['uniform', 'distance']}

model = GridSearchCV(estimator=pipeline_std_knn,              # estimator
             param_grid=search_space_grid,                    # the grid of the grid search
             scoring='f1',                                    # which measure to optimise on
             cv=2,                                            # number of folds during cross-validation
             n_jobs=-1)                                       # number of CPU cores to use (use -1 for all cores)

model.fit(features_train, target_train)
