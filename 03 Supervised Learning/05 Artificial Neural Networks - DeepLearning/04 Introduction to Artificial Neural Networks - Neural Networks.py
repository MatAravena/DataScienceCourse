# Learned what ANNs are and how they work.
# Understand the terms epoch and batch size.
# Generated an ANN yourself, using tensorflow.



# Now you can also take the results from a layer, to another layer, just like stacking. 
# This means the ANN is drawn out and the layers are connected. We distinguish between two types:
# Input layer: The first layer in an ANN is formed from the data set used. Each feature is assigned to a single artificial neuron. 
    # So this layer always has as many artificial neurons as there are features in the data set. This layer does not use any bias.
# Hidden layer: All layer between the input and output layers.




# # Artificial neural networks in tensorflow

# Sequential  is a submodule tensorflow.keras for building a feedforward ANN.
# it can be found in tensorflow.keras.models. Just like in sklearn, Sequential is instantiated and assigned to a variable.

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 

model_ann =Sequential()

# Dense(units = int,         # The dimensionality of the output space
#       activation = str,       # The activation function
#       input_dim = int/tuple # Number of dimensions of the features
#      )

# First we have units, which determine the number of artificial neurons in the layer. 
# Next we have activation, which defines the activation function. There is a large selection of built-in activation functions. We'll limit ourselves to the following two:

# activation = 'sigmoid': The Sigmoid function, from logistic regression. For binary classification problems, this is the first choice for the output layer.
# activation = 'relu': The Rectified Linear Unit function, ReLU for short, is a very good choice for hidden layers. It calculates the  Σ
#   value (see previous lesson) of the artificial neuron and the output becomes either  Σ
#   if  Σ
#   is greater than 0, or 0 if  Σ
#   is less than 0. It's a very simple function.

# The last argument is input_dim. This is only relevant to us in the first layer, the hidden layer with 35 artificial neurons. 
# This argument creates an input layer before the first layer. For this it's necessary to know how many dimensions the data set has. 
# In our case, it's the number of features. In more complicated cases, these can be images with several color channels, which have features along several axes, 
# called tensors (hence the name tensorflow).


hidden_layer = Dense(units = 35, activation = 'relu', input_dim = features_train.shape[1])
output_layer = Dense(units = 1, activation = 'sigmoid')

# Adding the layers to the Sequential model
model_ann.add(hidden_layer)
model_ann.add(output_layer)

# we have to convert it into machine code

model_ann.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])

# my_model.evaluate(my_feature_val,   # Validation feature data 
#                   my_target_val)    # Validation target vector

# It outputs a vector. Its first entry is the value of the loss function, and its second value is the value of the metric specified in the .compile() method, so in this case the accuracy

model_ann_eval = model_ann.evaluate(features_val_scaled, target_val)
model_ann_eval

# Remember:

# Instantiate the neural network with Sequential() and the corresponding layers with Dense().
# Add each layer one after the other with my_model.add(). Pay attention to the order the steps come in.
# The last layer generates the predictions, so you should specify the following parameters units=1 and activation='sigmoid'.
