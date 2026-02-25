import numpy as np

# tanh activation function
def tanh(x):
    return np.tanh(x)

# random weights between -0.5 and 0.5
w1 = np.random.uniform(-0.5, 0.5)
w2 = np.random.uniform(-0.5, 0.5)

# biases
b1 = 0.5
b2 = 0.7

# example input
x1 = 1
x2 = 2

# hidden layer
h = tanh(w1 * x1 + w2 * x2 + b1)

# output layer
output = tanh(h + b2)

print("Network Output:", output)
