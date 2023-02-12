import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline

# Importing the movies dataset (.csv file)
underrated_movies = pd.read_csv("../data/Underrated.csv")

# Training the neuron
underrated_movies["c_year"] = underrated_movies["Year"].apply(lambda x: 1 if x < 2000 else 0)
underrated_movies["c_genre"] = underrated_movies["Genres"].apply(lambda x: 1 if x == "Action" else 0)
underrated_movies["c_imdb_rating"] = underrated_movies["IMDb Rating"].apply(lambda x: 1 if x >= 6 else 0)
print(underrated_movies.head())

underrated_movies.plot(
    kind="scatter",
    x="c_year",
    y="c_genre",
    z="c_imdb_rating",
    colormap="jet"
)