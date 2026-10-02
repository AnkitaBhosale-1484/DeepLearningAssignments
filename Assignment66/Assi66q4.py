'''Write a Python program to show how weights are updated in ANN.
Tasks:
Take input, weight, bias, target output, and learning rate.
Calculate prediction.
Calculate error.
Update weight using gradient descent logic.
Display old weight and updated weight.'''

import math

# Input values
x1 = 2
x2 = 3

# Initial weights
w1 = 0.4
w2 = 0.6

# Bias
bias = 0.5

# Target output
target = 1

# Learning rate
learning_rate = 0.1

# Calculate weighted sum
weighted_sum = (x1 * w1) + (x2 * w2) + bias

# Apply sigmoid activation function
prediction = 1 / (1 + math.exp(-weighted_sum))

# Calculate error
error = target - prediction

# Calculate gradient
delta = error * prediction * (1 - prediction)

# Store old values
old_w1 = w1
old_w2 = w2
old_bias = bias

# Update weights
w1 = w1 + learning_rate * delta * x1
w2 = w2 + learning_rate * delta * x2

# Update bias
bias = bias + learning_rate * delta

# Display results
print("Prediction:", prediction)
print("Error:", error)

print("\nOld Weights:")
print("w1 =", old_w1)
print("w2 =", old_w2)
print("Bias =", old_bias)

print("\nUpdated Weights:")
print("w1 =", w1)
print("w2 =", w2)
print("Bias =", bias)