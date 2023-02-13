import pandas as pd

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