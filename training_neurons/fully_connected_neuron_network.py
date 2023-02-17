# Fully Connected Network example
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import colorbar, figure

from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from numpy import number
from tensorflow.python.training.input import batch

# Importing the dataset (.csv file)
traffic_data = pd.read_csv("../data/traffic_data.csv", index_col=0)

# Displaying the first 10 rows and the amount of deliveries with and without traffic
print(traffic_data.head(10))
print(traffic_data["type"].value_counts())

# Displaying the amount of each day of the week in the dataset
print(traffic_data["day"].value_counts())

# Displaying the amount of time stamps between each hour of the day in the dataset
# That would give us information about the times with most orders
print(traffic_data["hour"].value_counts())

# Displaying the data by day of the week (variables: traffic x no_traffic)
print(traffic_data.groupby("day")["type"].value_counts())

# Configuring traffic x no_traffic
traffic_data["c_type"] = traffic_data["type"].apply(lambda x: 1 if x == "traffic" else 0)

# Displaying the data on a graph
figure(num=None, figsize=(12, 10))
plt.scatter(
    traffic_data["day"],
    traffic_data["hour"],
    c=traffic_data["c_type"],
    cmap="jet", # jet applies a colormap
)
cbar = colorbar()
plt.show()

# Defining what is for training and what is for testing
training_dataset = traffic_data.sample(frac=0.8)
testing_dataset = traffic_data[~traffic_data.index.isin(training_dataset.index)]

# Selecting the columns
input_columns = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
    "hour",
    "minute",
    "second"
]

# Altering the dropout rate
traffic_model = Sequential([
    Dense(32, input_dim=len(input_columns), activation="relu"),
    Dropout(0.1),
    Dense(32, activation="relu"),
    Dropout(0.2),
    Dense(1, activation="sigmoid"),
])

adam = Adam()
traffic_model.compile(loss="binary_crossentropy", optimizer="adam",
                      metrics=["accuracy"])
traffic_model.summary()

# Training the model
batch_size = 100
history_traffic_model = traffic_model.fit(
    training_dataset[input_columns],
    training_dataset[["c_type"]],
    epochs=30,
    validation_split=0.1,
    batch_size=batch_size
)

# Evaluating the test data
test_loss, test_accuracy = traffic_model.evaluate(
    testing_dataset[input_columns],
    testing_dataset["c_type"]
)
print(f"Evaluation result on Test Data : Loss = {test_loss}, accuracy = {test_accuracy}")
# Loss:
# Accuracy: