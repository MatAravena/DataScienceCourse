# Evaluating Classification Performance


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



df_test = pd.read_csv('occupancy_test.txt')
df_test.loc[:, 'date'] = pd.to_datetime(df_test.loc[:, 'date'])
df_test.loc[:, 'Occupancy'] = df_test.loc[:, 'Occupancy'].astype('category')
df_test.loc[:, 'msm'] = (df_test.loc[:, 'date'].dt.hour * 60) + df_test.loc[:, 'date'].dt.minute


print(df_train.shape)
print(df_test.shape)


from sklearn.neighbors import KNeighborsClassifier 
from sklearn.preprocessing import StandardScaler

model =  KNeighborsClassifier(n_neighbors=3)
scaler = StandardScaler()

features_train = df_train.loc[:,['CO2','HumidityRatio']]
target_train= df_train.loc[:,'Occupancy']

features_train_standardized = scaler.fit_transform(features_train)

model.fit(features_train_standardized, target_train)


# Just like with a regression , you should evaluate the model quality with independent test data.
# This means pretending not to know the target vector in the test data and then comparing the predicted data with the actual data.



features_test = df_test.loc[:,['CO2','HumidityRatio']]
target_test   = df_test.loc[:,'Occupancy']

features_test_standardized = scaler.transform(features_test)

target_test_pred = model.predict(features_test_standardized)

# But because we are not predicting continuous values, the metrics you've already encountered, mean squared error and R², will not help us here. 
# These measure the similarity between measured and predicted numerical values. They are not useful for categories.

from sklearn.metrics import accuracy_score
accuracy_score(target_test, target_test_pred)
# 0.6626641651031895

# It achieved an accuracy of 66%. In other words: At two out of three points in time, 
# the model was able to detect the presence or absence of people in the room solely on the basis of CO2 values and the specific humidity.


# Confusion matrices
# Surprisingly often, classification models make use of these differences between categories and appear quite good. 
# However, in reality they are then no better than classifications based on a pie chart
#  order not to be fooled by the model, we can use other model quality metrics to determine accuracy.

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(target_test, target_test_pred)

recall = cm[1][1]/(cm[1][0] + cm[1][1])
precision = cm[1][1]/(cm[0][1] + cm[1][1])

# *F1 score*. This value is [the harmonic mean](https://en.wikipedia.org/wiki/Harmonic_mean) of *recall* and *precision*. The harmonic mean is a kind of average for relative frequencies.
# The harmonic mean is a kind of average for relative frequencies.

F1 = 2 * (precision * recall)/(precision + recall)




from sklearn.metrics import precision_score, recall_score, f1_score
print(recall_score(target_test, target_test_pred))
print(precision_score(target_test, target_test_pred))
print(f1_score(target_test, target_test_pred))


# You don't necessarily need your own test data set for this. Cross-validation also allows you 
# to create an independent validation data set from the training data. We'll look at that in the next lesson.

# Remember
# For unequal category sizes, the accuracy is not a good model quality metric.
# Recall and precision both focus on the cases that are actually positive.
# Recall and precision only differ in the population: either all cases that are actually positive (recall) or all positively predicted cases (precision)
# The F1 score is a robust metric that considers both false positives and false negatives