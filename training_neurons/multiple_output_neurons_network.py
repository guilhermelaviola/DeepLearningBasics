# Multiple output neuron network model trained with SGD
import pandas as pd
import matplotlib.pyplot as plt

from keras.layers import Dense
from keras.optimizers import SGD, Adam
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

# Training with SGD
# Configuring and training the neural network with multiple outputs
# with 15 input dimensions and 3 output neurons
pizza_type_model = Sequential()
pizza_type_model.add(Dense(3, input_dim=15, activation="softmax"))
sgd = SGD()
pizza_type_model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
pizza_type_model.summary()
# When we have multiple neurons producing binary values, we use the
# Categorical Crossentropy. The Binary Crossentropy is used when we
# have one single neuron producing a binary value

# Training the model
history_sgd_pizza_type_model = pizza_type_model.fit(
    training_dataset[["corn", "olives", "mushrooms", "spinach", "pineapple",
                      "artichoke", "chilli", "pepper", "onion", "mozzarella",
                      "egg", "pepperoni", "beef", "chicken", "bacon",]],
    training_dataset[["vegan", "vegetarian", "meaty"]],
    epochs=200,
    validation_split=0.2,
)
# The validation_split parameter keep part of the training data
# for validation

# Training with Adam
# Configuring and training the neural network with multiple outputs
# with 15 input dimensions and 3 output neurons
pizza_type_model = Sequential()
pizza_type_model.add(Dense(3, input_dim=15, activation="softmax"))
adam = Adam()
pizza_type_model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])
pizza_type_model.summary()

# Training the model
history_adam_pizza_type_model = pizza_type_model.fit(
    training_dataset[["corn", "olives", "mushrooms", "spinach", "pineapple",
                      "artichoke", "chilli", "pepper", "onion", "mozzarella",
                      "egg", "pepperoni", "beef", "chicken", "bacon",]],
    training_dataset[["vegan", "vegetarian", "meaty"]],
    epochs=200,
    validation_split=0.2,
)
# The validation_split parameter keep part of the training data
# for validation

# Plotting the comparison between both algorithms (ASG and Adam)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes = axes.flatten()

# Plotting SGD model training history
axes[0].plot(history_sgd_pizza_type_model.history["loss"])
axes[0].plot(history_sgd_pizza_type_model.history["accuracy"])
axes[0].set_title("SGD based Model Training History")
axes[0].set_ylabel("Value")
axes[0].set_xlabel("Epoch")
axes[0].legend(["Loss", "Accuracy"], loc="center-right")

# Plotting SGD model training history
axes[1].plot(history_sgd_pizza_type_model.history["loss"])
axes[1].plot(history_sgd_pizza_type_model.history["accuracy"])
axes[1].set_title("Adam based Model Training History")
axes[1].set_ylabel("Value")
axes[1].set_xlabel("Epoch")
axes[1].legend(["Loss", "Accuracy"], loc="center-right")

# Displaying the graphs
plt.plot()

# With Adam, the training can reach higer accuracy and lower loss more quicky than
# SGD, but Adam can lead to overfitting. What does that mean? Well, that means that
# although the training data have excellent results, the performance on testing data
# is worse. But Adam is widely used as Data Science algorithm