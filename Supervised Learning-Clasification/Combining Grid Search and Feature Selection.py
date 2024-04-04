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

# module import
from sklearn.exceptions import DataConversionWarning
import warnings

warnings.filterwarnings('ignore', category=DataConversionWarning)  # suppress data conversion warnings




from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

target_train = df_train.loc[:, 'Occupancy']
pipeline_std_knn = Pipeline([('std', StandardScaler()),
                             ('knn', KNeighborsClassifier())])
ks = [1, 2, 4, 7, 11, 19, 31, 51, 84, 138, 227, 372, 610, 1000]

search_space = {'knn__n_neighbors': ks,
                'knn__weights': ['uniform', 'distance']}

model = GridSearchCV(estimator = pipeline_std_knn, 
                     param_grid = search_space, 
                     scoring='f1',
                     cv=2,
                     n_jobs=-1)



# Testing feature combinations
import itertools
list(itertools.combinations(col_of_interest, 2))

feature_combinations = []
for possible_size_of_combinations in range(1,  len(col_of_interest) + 1):  # for each number of items to be combined

    new_combinations = list(itertools.combinations(col_of_interest, 
                                                   possible_size_of_combinations))
    feature_combinations += new_combinations

feature_combinations


# Testing the features combinations

for cols in feature_combinations:  # for each feature combination
    print('Features: ', cols)
    features_train = df_train.loc[:, cols]
    model.fit(features_train, target_train)
    print('Best F1-score: ', round(model.best_score_, 3))
    print('Model spec: ', model.best_estimator_, '\n\n')


# Predicting test data
    
# instantiate model
model = KNeighborsClassifier(n_neighbors=1000, weights='distance')

# define features matrix
features_train = df_train.loc[:, ['Humidity', 'Light', 'HumidityRatio']]

# standardize features matrix
standardizer = StandardScaler()
standardizer.fit(df_train.loc[:, ['Humidity', 'Light', 'HumidityRatio']])
features_train_standardized = standardizer.transform(features_train)

# train model
model.fit(features_train_standardized, target_train)


# Testing data
df_test = pd.read_csv('occupancy_test.txt')

# turn date into DateTime (not really necessary)
df_test.loc[:, 'date'] = pd.to_datetime(df_test.loc[:, 'date'])

# turn Occupancy into category (not really necessary)
df_test.loc[:, 'Occupancy'] = df_test.loc[:, 'Occupancy'].astype('category')

# define new feature (not really necessary)
df_test.loc[:, 'msm'] = (df_test.loc[:, 'date'].dt.hour * 60) + df_test.loc[:, 'date'].dt.minute

# features matrix and target vector
features_test = df_test.loc[:, ['Humidity', 'Light', 'HumidityRatio']]
target_test = df_test.loc[:, 'Occupancy']

# standardize features matrix
features_test_standardized = standardizer.transform(features_test)



# Check the real values
from sklearn.metrics import f1_score

target_test_pred = model.predict(features_test_standardized)

f1_score(target_test, target_test_pred)


# Remember:

# Combine list entries with itertools.combinations()
# Feature selection together with grid searches takes a lot of time
# You should always evaluate the best model with independent test data at the end