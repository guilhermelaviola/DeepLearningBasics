import pandas as pd
import matplotlib.pyplot as plt

from keras.layers import Dense
from keras.optimizers import SGD
from keras.models import Sequential

# Importing and displaying the dataset (.csv file)
pizza_types = pd.read_csv("../data/pizza_types.csv", index_col=0)
print(pizza_types.head())

# We have 5520 lines in this dataset. Generally, we use 80% of the data
# for training and the other 20% we use for testing
# In this case, 80% would be represented by 4416 rows
training_dataset = pizza_types.sample(frac=0.8)
# The ~ means NOT, so here we have the remaining elements: 1104 rows
testing_dataset = pizza_types[~pizza_types.index.isin(training_dataset.index)]

# Displaying both dataset shapes
print("training_dataset shape:", training_dataset.shape)
print("testing_dataset shape:", testing_dataset.shape)

# Configuring and training the neural network with multiple outputs
# with 15 input dimensions and 3 output neurons
pizza_type_model = Sequential()
pizza_type_model.add(Dense(3, input_dim=15, activation="softmax"))
sgd = SGD()
pizza_type_model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
pizza_type_model.summary()
# When we have multiple neurons producing binary values, we use the
# Categorical Crossentropy. The Binary Crossentropy is used when we
# have one single neuron producing a binary value