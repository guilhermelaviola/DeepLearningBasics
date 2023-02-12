import tensorflow
from keras.layers import Dense
from keras.optimizers import SGD
from keras.models import Sequential
from identifier import Identifier
from identifier.Identifier import underrated_movies

# Configuring units (amount of neurons), dimensions (categories),
# the function the neuron will be using and also its loss function
# in this case, we're using Binary Cross Entropy => loss = yi * log(pi) + (1 - yi) * log(1 - p1)
single_neuron_layer = Dense(
    units=1,
    input_dim=2,
    activation="sigmoid",
    loss="binary_crossentropy"
)
# Stochastic Gradient Descent - SGD)
sgd = SGD()
# Configuring the model in order to make the layers connect to each other sequentially
single_neuron_model = Sequential()
identifier = Identifier()

# Importing all components and checking the configuration with summary()
single_neuron_model.add(single_neuron_layer)
single_neuron_model.compile(loss=loss, optimizer=sgd, metrics=["accuracy"])
single_neuron_model.summary()

# Doing the training
df = identifier.underrated_movies
history = single_neuron_model.fit(
    df[["c_year", "c_gender"]].values,
    df[["c_imdb_rating"]].values,
    epochs=2500)

# Observing the dataset previsions which is being executed by the model
test_loss, test_accuracy = single_neuron_model.evaluate(
    underrated_movies[["c_year", "c_gender"]],
    underrated_movies["c_imdb_rating"],
print(f"Evaluation result on Test Data : Loss = {test_loss}, Accuracy = {test_accuracy}"))

# Altough the loss in this model is not = 0, its accuraccy reached 1,0 = 100%
# So it's a successful model for the use case