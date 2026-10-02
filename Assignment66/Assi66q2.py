#Write a Python program to demonstrate different activation functions.
#Functions to implement:
#Sigmoid
#ReLU
#Tanh
#Tasks:
#Accept input values from -10 to 10.
#Plot all activation functions using Matplotlib.
#Explain the use of each activation function.

import numpy as np
import matplotlib.pyplot as plt

# Input values from -10 to 10
x = np.linspace(-10, 10, 100)

# Sigmoid activation function
sigmoid = 1 / (1 + np.exp(-x))

# ReLU activation function
relu = np.maximum(0, x)

# Tanh activation function
tanh = np.tanh(x)

# Plot Sigmoid
plt.figure()
plt.plot(x, sigmoid)
plt.title("Sigmoid Activation Function")
plt.xlabel("Input")
plt.ylabel("Output")
plt.grid()
plt.show()

# Plot ReLU
plt.figure()
plt.plot(x, relu)
plt.title("ReLU Activation Function")
plt.xlabel("Input")
plt.ylabel("Output")
plt.grid()
plt.show()

# Plot Tanh
plt.figure()
plt.plot(x, tanh)
plt.title("Tanh Activation Function")
plt.xlabel("Input")
plt.ylabel("Output")
plt.grid()
plt.show()