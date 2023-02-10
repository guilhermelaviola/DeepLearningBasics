import tensorflow
from keras.layers import Dense

# Configuring units (amount of neurons), dimensions (categories) and the function the neuron will be using
single_neuron_layer = Dense(
    units=1,
    input_dim=2,
    activation="sigmoid"
)