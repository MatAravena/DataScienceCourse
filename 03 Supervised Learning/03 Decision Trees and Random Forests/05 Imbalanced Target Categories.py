# Reduce the number of majority class data points (typically a random sample)
# Artificially increase the number of datapoints in the minority class (typically using what's called bootstrapping)
# Give more weight to the incorrect classification of minority data points when training the model



# Reducing the size of the majority category or reducing the size of the minority category

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



# Resampling strategies
# Undersampling: reducing the number of majority class data points (typically a random sample).
from imblearn.under_sampling import RandomUnderSampler

#1. initialize
undersampler = RandomUnderSampler(random_state=42)

#2. define Features and Target
# see first cell of notebook

#3. fit and resample
features_under, target_under = undersampler.fit_resample(features_train, target_train)


# Therefore, we only recommend this strategy if you have an extremely large training data set and the smaller category is not too small.




# Oversampling: artificially increasing the number of datapoints in the minority class
from imblearn.over_sampling import RandomOverSampler

#1. initiatiate
oversampler = RandomOverSampler(random_state=42)

#2. fit and resample
features_over, target_over = oversampler.fit_resample(features_train, target_train)

#show classes
pd.crosstab(index=target_over, columns='count')

# oversampling is to artificially create data points that resemble
# SMOTE : This name is an acronym and stands for Synthetic Minority Oversampling TEchnique
from imblearn.over_sampling import SMOTE
smotesampler = SMOTE()
smotesampler.fit(features_train,target_train)

features_smote, target_smote = smotesampler.fit_resample(features_train,target_train)

pd.crosstab(index=target_smote, columns='count')



print('Oversampling:', features_over.drop_duplicates(keep=False).shape)
print('SMOTE:', features_smote.drop_duplicates(keep=False).shape)
# Oversampling: (865, 15)
# SMOTE: (1724, 15)



# Give more weight to the incorrect classification of minority data points when training the model.
from sklearn.tree import DecisionTreeClassifier
model_unbalanced = DecisionTreeClassifier(random_state=42, max_depth=12)
model_balanced_by_class_weights = DecisionTreeClassifier(random_state=42, max_depth=12,class_weight='balanced')


# **Congratulations:** You have learned four approaches to compensate for size differences in the target vector categories. Next, we'll look at how well these approaches perform by calculating the model quality measures from the test data.


# Evaluating compensation approaches
# So the right time for resampling with cross-validation is after the training-validation split and should only be carried out 
# on the data which the model actually trains with.

from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from imblearn.pipeline import Pipeline #our new pipeline builer

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score


tree_clf = DecisionTreeClassifier(random_state=42)
search_space = {'estimator__max_depth': range(2, 16, 2), 'estimator__class_weight': [None, 'balanced']}

# 'passthrough', which both sklearn pipelines and the imblearn interpret "skip this step
samplers = [('oversampling', oversampler),
            ('undersampling', undersampler),
            ('class_weights', 'passthrough'),
            ('SMOTE', smotesampler)]

# storage container for results
results = []

# go through every sampler
for name, sampler in samplers:
    #sampling
    imb_pipe = Pipeline([('sampler', sampler),
                         ('estimator', tree_clf)])

    #gridsearch and CV
    grid = GridSearchCV(estimator=imb_pipe,
                        param_grid=search_space,
                        n_jobs=-1,
                        cv=5,
                        scoring='f1')

    grid.fit(features_train, target_train)

    #evaluation
    model = grid.best_estimator_.named_steps['estimator']
    recall = recall_score(target_test, model.predict(features_test))
    precision = precision_score(target_test, model.predict(features_test))

    #verbose
    print(name.upper())
    print(grid.scoring, 'on Validationset:', grid.best_score_ )
    print("precision :", precision)
    print("recall :", recall)
    print(model)
    print('#'*11)

    #save
    scores = {'name': name,
              'precision': precision,
              'recall': recall}
    results.append(scores)

#show results
pd.DataFrame(results)

# Balancing the 'attrition' categories during the model fitting using the class_weights parameter results in the second best precision and second best recall.
# You should be extremely careful with the extremely good recall value undersampling receives. In this case, the training data set is so small that we can expect
# these values to depend heavily on which data is used in the training data. The model with the SMOTE resampling strategy has the greatest relevance, but it also has the worst recall.

# Important: You can find the class_weights parameter in almost all sklearn classification algorithms. Unfortunately, the most important exception is NeighborsClassifier, which currently doesn't offer this functionality. It's generally a good idea to set class_weights to class_weight='balanced', even if the differences between the target vector categories are not that big. This ensures that even the smaller category, whose classification is typically of particular interest, is well predicted.

# Congratulations: You have learned how to use the class_weights parameter to train classification models so that both target vector categories are equally important. Alternatively, you can reduce the larger category or increase the smaller category by duplicating or inserting synthetic data (SMOTE). But proceed with caution - especially with undersampling. Usually we recommend the solution with class_weights or SMOTE.

# But thankfully you now know the bootstrap procedure, which we can use in the next lesson to progress from decision trees to decision forests, known as random forests. Random forests are a very popular classification approach. We'll look at them in the next lesson.


# Remember:

# The bootstrap procedure is like taking a data point and putting it back so it can be taken again.
# For classification algorithms in general, specify class_weight='balanced' or resample the data with SMOTE.
# With undersampling and oversampling the number of data points in the classes is adjusted.

