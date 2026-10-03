'''Demonstrate ReLU and Max Pooling
Given feature map
3   3   3
0   0   0
-3 -3  -3
The assignment asks us to:
Create feature map with positive and negative values.
Apply ReLU.
Apply 2×2 max pooling.
Display output after each step.
Explain why pooling reduces size.
ReLU applies max(0, x), so negative values become 0 while positive values remain unchanged. Max pooling keeps the maximum value from each pooling region and reduces spatial dimensions'''


#  ReLU and Max Pooling

feature_map = [
    [3, 3, 3],
    [0, 0, 0],
    [-3, -3, -3]
]

# ReLU function
def relu(value):
    if value < 0:
        return 0
    else:
        return value

# Apply ReLU
relu_output = []

for row in feature_map:
    new_row = []

    for value in row:
        new_row.append(relu(value))

    relu_output.append(new_row)

print("ReLU Output:")
for row in relu_output:
    print(row)


# 2x2 Max Pooling with stride 2
pool_size = 2
max_pool_output = []

for i in range(0, len(relu_output) - 1, 2):
    row = []

    for j in range(0, len(relu_output[0]) - 1, 2):

        values = [
            relu_output[i][j],
            relu_output[i][j + 1],
            relu_output[i + 1][j],
            relu_output[i + 1][j + 1]
        ]

        row.append(max(values))

    max_pool_output.append(row)

print("\nMax Pooling Output:")
for row in max_pool_output:
    print(row)