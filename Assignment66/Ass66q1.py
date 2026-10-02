#Simulate a Single Artificial Neuron
#Write a Python program to simulate a single artificial neuron.
#Input:
#x1 = 2
#x2 = 3
#w1 = 0.4
#w2 = 0.6
#bias = 0.5

import math

# Input values
x1 = 2
x2 = 3

# Weights
w1 = 0.4
w2 = 0.6

# Bias
bias = 0.5

# Calculate weighted sum
weighted_sum = (x1 * w1) + (x2 * w2) + bias

# Sigmoid activation function
output = 1 / (1 + math.exp(-weighted_sum))

print("Weighted Sum:", weighted_sum)
print("Sigmoid Output:", output)

# Check whether output is close to 0 or 1
if output >= 0.5:
    print("Output is close to 1")
else:
    print("Output is close to 0")