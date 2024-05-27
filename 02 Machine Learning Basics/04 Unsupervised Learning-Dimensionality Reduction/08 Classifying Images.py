# Speeding up training with PCA

# the data is in the database letters.db 
import sqlalchemy as sa
engine = sa.create_engine('sqlite:///letters.db')  # connect to the database
connection = engine.connect()  # create the engine

# import pandas and read the data and the labels
import pandas as pd
df_img = pd.read_sql('SELECT * FROM images', con=connection).drop('index', axis=1)
df_labels = pd.read_sql('SELECT * FROM labels', con=connection)

connection.close() # close connection to database

# import and instantiate the scaler and pca 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
scaler = StandardScaler()
pca = PCA(n_components=0.98, random_state=10)  # keep 98 percent of the variance

arr_img_std = scaler.fit_transform(df_img)
arr_img_std_pca = pca.fit_transform(arr_img_std)



# We'll try it out now. For this purpose we'll consider our letters as a classification problem.
# We want to know how well we can determine whether the image section is a C or an I.
from sklearn.neighbors import KNeighborsClassifier 
model = KNeighborsClassifier(n_neighbors=10)

%%timeit -n 10 -r 3 
model.fit(arr_img_std, df_labels.loc[:, 'labels'])

%timeit -n 10  -r 3 model.fit(arr_img_std_pca, df_labels.loc[:, 'labels'])



from sklearn.pipeline import Pipeline

# without pca
pipe = Pipeline([('std', scaler),
                 ('knn', model)])

# with pca
pipe_pca = Pipeline([('std', scaler),
                    ('pca', pca),
                    ('knn', model)])

from sklearn.model_selection import GridSearchCV
search_space = {'knn__n_neighbors': [1, 2, 3, 4, 5, 10]}

from sklearn.model_selection import KFold
kf = KFold(n_splits=3, shuffle=True, random_state=10)

neighbor_search = GridSearchCV(estimator=pipe,                # estimator
             param_grid=search_space,                         # the grid of the grid search
             scoring='f1',                                    # which measure to optimise on
             cv=kf,                                           # number of folds during cross-validation
             n_jobs=-1 )                                      # number of CPU cores to use (use -1 for all cores)

# As GridSearchCV can not handle properly letters as categories in Datasets or Series
# This property transform the data into numerical values as is explnained in the following code
from sklearn.preprocessing import LabelEncoder

label = LabelEncoder()
target = label.fit_transform(df_labels.loc[:, 'labels'])
print(target)
# [0 0 0 ... 1 1 1]



# No comparing the models with or without PCA

neighbor_search.fit(df_img, target)
print(neighbor_search.best_score_)
print(neighbor_search.best_params_)
# 0.9716419001280641
# {'knn__n_neighbors': 3}

neighbor_search_pca = GridSearchCV(estimator=pipe_pca,
                                   param_grid=search_space,
                                   scoring='f1',
                                   cv=kf,
                                   n_jobs=-1,
                                   verbose=4)

neighbor_search_pca.fit(df_img, target)
print(neighbor_search_pca.best_score_)
print(neighbor_search_pca.best_params_)

# 0.9732774695638415
# {'knn__n_neighbors': 3}


# Remember:
# The fewer features the data has, the faster you can train your model and make predictions
# Measure the execution time of a cell with %%timeit