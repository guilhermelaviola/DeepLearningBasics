# Plotting a Sigmoid function with matplotlib
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

values = np.arange(-10, 10, 0.1)
print([sigmoid(value) for value in values])

# Graphic values calculated
plt.plot(values, sigmoid(values))
plt.xlabel('X axis')
plt.ylabel('Sigmoid(X)')
plt.title('Sigmoid function in Matplotlib')
plt.show()