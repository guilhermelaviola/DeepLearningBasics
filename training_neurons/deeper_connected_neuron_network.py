import pandas as pd
import matplotlib.pyplot as plt

from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD

# Importing the dataset (.csv file)
traffic_data = pd.read_csv("../data/traffic_data.csv", index_col=0)

# Displaying the first 10 rows and the amount of deliveries with and without traffic
print(traffic_data.head(10))
traffic_data["type"].value_counts()

# Displaying the amount of each day of the week in the dataset
traffic_data["day"].value_counts()

# Displaying the amount of time stamps between each hour of the day in the dataset
# That would give us information about the times with most orders
traffic_data["day"].value_counts()

# Displaying the data by day of the week (variables: traffic x no traffic)
traffic_data.groupby("day")["type"].value_counts()