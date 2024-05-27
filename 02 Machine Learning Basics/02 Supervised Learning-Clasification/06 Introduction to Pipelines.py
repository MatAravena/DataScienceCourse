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



# pipelines are a sequence of data processing steps. A pipeline summarizes steps that would otherwise take many lines of code.
# They also have the huge advantage that you can pass them to functions such as cross_val_score() to perform cross-validation quickly.


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

features_train = df_train.loc[:, ['CO2', 'msm']]
target_train  = df_train.loc[:, 'Occupancy']

#ignore warnings
from sklearn.exceptions import DataConversionWarning
import warnings
warnings.filterwarnings(action='ignore', category=DataConversionWarning)



scaler = StandardScaler()  # instantiate scaler
scaler.fit_transform(features_train)  # train scaler and transform features matrix

# You can do the same in just one step by using a pipeline.

pipeline_std = Pipeline([('std', StandardScaler())])  # instantiate pipeline with one step (scaler)
pipeline_std.fit_transform(features_train)  # train pipeline (scaler) and transform features matrix




# Compare
from sklearn.neighbors import KNeighborsClassifier
pipeline_std_knn = Pipeline([('std', StandardScaler()), 
                             ('knn', KNeighborsClassifier(n_neighbors=3))])
pipeline_std_knn.fit(features_train, target_train)

# and this
model_knn = KNeighborsClassifier(n_neighbors=3)  # instantiate k-Nearest-Neighbors model
model_knn.fit(scaler.fit_transform(features_train),  # model fitting with standardised features matrix
              target_train)


# You can now use pipeline_std_knn and model_knn exactly the same way. They would make the same predictions.


# Cross validation with cross_val_score()
from sklearn.model_selection import cross_val_score

cv_results = cross_val_score(estimator=pipeline_std_knn,#pipeline or Model
                            X=features_train,           #feature matrix
                            y=target_train,             #target values
                            cv=2,                       #number of folds for cross-validation
                            scoring='f1',               #scoring function e.g ['accuracy','f1','recall']
                            n_jobs=-1)                  #number of cpu cores to use for computation (use -1 to use all available cores)
cv_results


# Remember:

# You can chain together transformers like Standard Scaler, potentially with an estimator like NeighborsClassifier
# Pipeline expects a list of ('name', transformer or estimator) pairs
# cross_val_score() performs cross validation in one line of code