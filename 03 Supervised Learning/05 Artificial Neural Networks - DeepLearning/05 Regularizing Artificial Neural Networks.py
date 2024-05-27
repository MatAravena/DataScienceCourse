# How to identify overfitting in ANNs.
# What a dropout layer is.
# What the term early stopping means.


# read data
import pandas as pd
features_train = pd.read_pickle('features_train.p')
features_val = pd.read_pickle('features_val.p')
target_train = pd.read_pickle('target_train.p')
target_val = pd.read_pickle('target_val.p')

# scale data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
features_train_scaled = scaler.fit_transform(features_train)
features_val_scaled = scaler.transform(features_val)

features_train.head()


# Artificial neural networks are multilayer neural networks if they have at least one hidden layer. 
# If they have one hidden layer, we also refer to them as shallow neural networks. 
# Theoretically, one hidden layer is already enough to create very complex functions. 
# But in practice, a neural network is easier to train if it has several hidden layers. 
# Then we call this a deep neural network and the term deep learning.

from tensorflow.keras import Sequential
model_ann = Sequential()

from tensorflow.keras.layers import Dense

hidden_first   = Dense(units = 5, activation = 'relu', input_dim = features_train.shape[1])
hidden_second  = Dense(units = 5, activation = 'relu')
hidden_third   = Dense(units = 5, activation = 'relu')
hidden_fourth  = Dense(units = 5, activation = 'relu')
hidden_fifth   = Dense(units = 5, activation = 'relu')

model_ann.add(hidden_first)
model_ann.add(hidden_second)
model_ann.add(hidden_third)
model_ann.add(hidden_fourth)
model_ann.add(hidden_fifth)


output_layer = Dense(units = 5, activation = 'sigmoid')
model_ann.add(output_layer)

model_ann.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])


# Import plot_model
from tensorflow.keras.utils import plot_model
# plot model
plot_model(model_ann)


hist_ann = model_ann.fit(features_train_scaled,
                         target_train,
                         epochs=20,
                         batch_size=64,
                         validation_data=(features_val_scaled, target_val))

# Print the model history
print(hist_ann.history)



# # Limiting overfitting
# Dropout a way to do the regularization for ANN

# In this process, some randomly selected artificial neurons are deactivated at each learning step of the ANN. 
# So they only return the value 0 and their weights are not adjusted in this step. 
# Overfitting is when the model learns the data by heart and some weights are very strongly defined. 
# But this doesn't happen as much if some neurons are left out As a result, the architecture of the neural network
# looks a little different at each learning step. This reduces the dependencies between the neurons. 
# The entire network becomes more robust and can be better generalized to to deal with data it hasn't yet seen. 
# You should use the dropout method particularly when the training set is relatively small.

from tensorflow.keras.layers import Dropout
model_ann_drop = Sequential()
dropout_layer_first = Dropout(rate=0.3)
dropout_layer_second = Dropout(rate=0.3)
dropout_layer_third = Dropout(rate=0.3)
dropout_layer_fourth = Dropout(rate=0.3)
dropout_layer_fifth = Dropout(rate=0.3)

model_ann_drop.add(hidden_first)
model_ann_drop.add(dropout_layer_first)

model_ann_drop.add(hidden_second)
model_ann_drop.add(dropout_layer_second)

model_ann_drop.add(hidden_third)
model_ann_drop.add(dropout_layer_third)

model_ann_drop.add(hidden_fourth)
model_ann_drop.add(dropout_layer_fourth)

model_ann_drop.add(hidden_fifth)
model_ann_drop.add(dropout_layer_fifth)

# compile
model_ann_drop.compile(optimizer = "adam", loss = 'binary_crossentropy', metrics = ['accuracy'])
plot_model(model_ann_drop)


hist_ann_drop = model_ann_drop.fit(features_train_scaled, 
                                   target_train, 
                                   epochs=20, 
                                   batch_size=64, 
                                   validation_data=(features_val_scaled, target_val))

# In contrast to sklearn, keras doesn't fit the model to the data starting from scratch, but it continues where you left off
hist_ann_drop = model_ann_drop.fit(features_train_scaled, target_train, epochs=10, batch_size=64, validation_data=(features_val_scaled, target_val))



# Another way to avoid overfitting the model too much is to stop training early. This procedure is called **early stopping**.
# define model
model_ann_early = Sequential()

# define hidden layers
hidden_first = Dense(units=50, activation='relu', input_dim=features_train_scaled.shape[1])
hidden_second = Dense(units=50, activation='relu')
hidden_third = Dense(units=50, activation='relu')
hidden_fourth = Dense(units=50, activation='relu')
hidden_fifth = Dense(units=50, activation='relu')

# define output layer
output_layer  = Dense(units=1, activation='sigmoid')

# add 5 hidden layers with 50 units
model_ann_early.add(hidden_first)
model_ann_early.add(hidden_second)
model_ann_early.add(hidden_third)
model_ann_early.add(hidden_fourth)
model_ann_early.add(hidden_fifth)

# add output layer
model_ann_early.add(output_layer)

# compile model
model_ann_early.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# plot the model
plot_model(model_ann_early)


# method so that they are implemented during training.  --> EarlyStopping
# It stops training if a specified metric doesn't continue to improve. 
# EarlyStopping has to be instantiated. 
# You pass the name of the value to be monitored to the monitor parameter as accuracy of the validation data 'val_accuracy'. 
# The parameter min_delta indicates the minimum size of an improvement that has to be achieved in order to be considered an improvement. 
# During training we could see that 'val_accuracy' was fluctuating in the third decimal place. '0.001'
# patience indicates how many epochs without improvement can happen before the training is stopped. '2' 



from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping( monitor='val_accuracy', min_delta=0.001, patience=2)

# train model using early stopping
model_ann_early.fit(features_train_scaled, target_train,
              epochs=200, batch_size=64,
              callbacks=[early_stop],
              validation_data=(features_val_scaled, target_val))


# HDF5 file
model_ann_early.save('model_ann.h5')

# tensorflow.keras.models offers the load_model function, which only requires the model's file path.



# Remember:

# Creating and training deep neural networks can be very time consuming.
# You can use the training history to evaluate if your model is overfitted.
# EarlyStopping stops the training and can save a lot of computing time.
# Dropout regularizes the ANN and makes it more robust.