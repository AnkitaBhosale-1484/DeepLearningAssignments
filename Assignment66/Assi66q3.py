'''Write a Python program to calculate loss manually.
Tasks:
Implement Mean Squared Error.
Implement Binary Cross Entropy.
Take actual and predicted values.
Display the calculated loss.
Explain which loss function is used for regression and classification.'''


import math

# Actual values
actual = [1, 0, 1, 1]

# Predicted values
predicted = [0.8, 0.2, 0.6, 0.9]

# Calculate Mean Squared Error
mse = 0

for i in range(len(actual)):
    mse += (actual[i] - predicted[i]) ** 2

mse = mse / len(actual)

# Calculate Binary Cross Entropy
bce = 0

for i in range(len(actual)):
    bce += -(actual[i] * math.log(predicted[i]) +
             (1 - actual[i]) * math.log(1 - predicted[i]))

bce = bce / len(actual)

# Display results
print("Actual Values:", actual)
print("Predicted Values:", predicted)
print("Mean Squared Error:", mse)
print("Binary Cross Entropy:", bce)