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




from sklearn.model_selection import KFold
kf = KFold(n_splits=2)
type(kf)
kf.split(df_train)
list(kf.split(df_train))



from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.preprocessing import StandardScaler


model = KNeighborsClassifier(n_neighbors=3)

scaler = StandardScaler()


step = 0
for train_index, val_index in kf.split(df_train):  # for each fold
    step = step + 1
    print('Step ', step)

    features_fold_train = df_train.iloc[train_index, [4, 5]]  # features matrix of training data (of this step)
    features_fold_val = df_train.iloc[val_index, [4, 5]]  # features matrix of validation data (of this step)   
    
    target_fold_train = df_train.iloc[train_index, 6]  # target vector of training data (of this step)
    target_fold_val = df_train.iloc[val_index, 6]  # target vector of validation data (of this step)
    

    scaler.fit(features_fold_train)
    features_fold_train_standardized = scaler.transform(features_fold_train)
    features_fold_val_standardized = scaler.transform(features_fold_val)

    model.fit(features_fold_train_standardized, target_fold_train)
    target_fold_val_pred = model.predict(features_fold_val_standardized)

    print('recall :',    recall_score(target_fold_val, target_fold_val_pred))
    print('precision :', precision_score(target_fold_val, target_fold_val_pred))
    print('f1 :',        f1_score(target_fold_val, target_fold_val_pred))
    print('/n')

# Step  1
# recall : 0.49605609114811566
# precision : 0.771117166212534
# f1 : 0.6037333333333332


# Step  2
# recall : 0.8996598639455783
# precision : 0.66125
# f1 : 0.7622478386167146
    
# What do these values express? 70% of the instances with people in the room were identified as such.
# 72% of positive predictions (a person is supposedly in the room) are correct. To summarize, the prediction quality in the positive cases (person in the room) is 68%.
    

# Remember:

# Evaluating models with cross validation prevents overfitting
# Carry out k-fold cross validation with KFold
# You can freely choose the number of folds of the training data set between 2 and the number of data points in the dataset