import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD

# Creating a dataset
bad_pizza_dataset = pd.DataFrame.from_dict({
    "tomato_sauce": ["no", "no", "yes", "yes"],
    "barbecue_sauce": ["no", "yes", "no", "yes"],
    "result": ["sauce error", "good", "good", "sauce error"]})

# Creating new columns to represent the 'problems' with numbers
bad_pizza_dataset["c_tomato_sauce"] = bad_pizza_dataset["tomato_sauce"].apply(lambda x: 1 if x == "yes" else 0)
bad_pizza_dataset["c_barbecue_sauce"] = bad_pizza_dataset["barbecue_sauce"].apply(lambda x: 1 if x == "yes" else 0)
bad_pizza_dataset["c_result"] = bad_pizza_dataset["result"].apply(lambda x: 0 if x == "sauce error" else 0)

# Displaying the dataset
print(bad_pizza_dataset.to_string())

# Plotting the graph
bad_pizza_dataset.plot(
    kind="scatter",
    x="c_tomato_sauce",
    y="c_barbecue_sauce",
    c="c_result",
    colormap="jet"
)

plt.show()

#Configuring and training a simple neural network
# Configuring the layers
input_layer = Dense(units=2, input_dim=2, activation="sigmoid")
output_layer = Dense(units=1, activation="sigmoid")

# Configuring the model
bad_pizza_model = Sequential()
bad_pizza_model.add(input_layer)
bad_pizza_model.add(output_layer)
sgd = SGD()
bad_pizza_model.compile(loss="binary_crossentropy", optimizer=sgd, metrics=["accuracy"])

# Displaying the configuration
# It shows 9 trainable parameters
bad_pizza_model.summary()

# Training the model
# The 'epoch' parameter indicates how many times the network
# must calculate the training data
history = bad_pizza_model.fit(
    bad_pizza_dataset[["c_tomato_sauce", "c_barbecue_sauce"]],
    bad_pizza_dataset["c_result"],
    epochs=3000,
)

# Finding out the model performance
test_loss, test_accuracy = bad_pizza_model.evaluate(
    bad_pizza_dataset[["c_tomato_sauce", "c_barbecue_sauce"]],
    bad_pizza_dataset["c_result"]
)
print(f"Evaluation result on Test Data : Loss = {test_loss}, accuracy = {test_accuracy}")