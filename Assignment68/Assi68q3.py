'''Write a Python program to show Flattening
Given matrix
6  4
8  6
Flattening converts a 2D feature map/matrix into a 1D vector so it can be passed to a fully connected/dense layer'''

#  Flattening

matrix = [
    [6, 4],
    [8, 6]
]

# Flatten 2D matrix into 1D vector
flatten_output = []

for row in matrix:
    for value in row:
        flatten_output.append(value)

print("Input Matrix:")
for row in matrix:
    print(row)

print("\nFlatten Output:")
print(flatten_output)


# Fully Connected Layer
weights = [0.1, 0.2, 0.3, 0.4]
bias = 0.5

output = 0

for i in range(len(flatten_output)):
    output = output + flatten_output[i] * weights[i]

output = output + bias

print("\nFinal Output:")
print(output)