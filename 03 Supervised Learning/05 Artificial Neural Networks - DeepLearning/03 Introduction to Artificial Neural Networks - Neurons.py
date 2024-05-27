# What artificial neurons are and how they work.
# How to put artificial neurons into practice.
# # What is the relationship between logistic regression and artificial neurons.



# # Artificial neurons

import pandas as pd
features_train = pd.read_pickle('features_train.p')
features_val = pd.read_pickle('features_val.p')
target_train = pd.read_pickle('target_train.p')
target_val = pd.read_pickle('target_val.p')

features_train.head()
print(target_train.value_counts(normalize=True))
print(target_val.value_counts(normalize=True))





# As a machine learning model, the artificial neuron will learn its weights ${w}$ based on the data. 
# In supervised learning, the predictions are compared with true values and in unsupervised learning, 
# the predictions are grouped by similarity. The artificial neuron, like all other machine learning models, uses the gradient method to determine the best weights here.

# **Remember:** A single artificial neuron is a generalization of logistic regression!
# The artificial neuron is a generalization because the activation function can be selected in many ways, whereas in logistic regression it is always the sigmoid function.



# # Artificial neurons versus logistic regression

# we want to verify the claim that logistic regression is the same model as an artificial neuron with the *sigmoid* activation function

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(features_train)
features_train_scaled = scaler.transform(features_train)
features_val_scaled = scaler.transform(features_val)

import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)



# Tensorflow with keras
# Keras been the library as sklearn

#!!!! IMPORTANT Only Tensorflow with a higer version than 2.X allow to user keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model_an = Sequential()  # define the model type
model_an.add(Dense(1, activation='sigmoid', input_dim=features_train_scaled.shape[1]))  # add one artificial neuron with sigmoid activtaion function
model_an.compile(optimizer = "adam", loss = 'binary_crossentropy', metrics = ['accuracy'])  # compile the model

model_an.fit(features_train_scaled, target_train, epochs=5) # Fit the model

target_val_pred_an = model_an.predict(features_val_scaled)

# In contrast to `LogisticRegression`, the `keras` model outputs a nested array with the results of the activation function. 
# First we'll convert this into a 1-dimensional array. We can achieve this with the `my_array.flatten()` method.
target_val_pred_an = target_val_pred_an.flatten()


# The values False and True are interpreted by Accuracy_score as 0 and 1.
target_val_pred_an = target_val_pred_an > 0.5


accuracy_score(target_val, target_val_pred_an)

# TensorFlow already evaluate it without need to predict and then calcualte the accuracy
model_an.evaluate(features_val_scaled, target_val) # Evaluate the model


# Remember:

# Artificial neurons can be implemented quickly with tensorflow in a similar way to sklearn.
# An artificial neuron assigns weights and a bias to the features and transforms the data with an activation function.
# An artificial neuron is a generalization of logistic regression!
