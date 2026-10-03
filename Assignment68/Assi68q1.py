'''Write a Python program to manually perform convolution
Given
Input image: 5 × 5
image = [
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
]
Kernel: 3 × 3 edge detection filter
kernel = [
    [-1, -1, -1],
    [ 0,  0,  0],
    [ 1,  1,  1]
]
Since there is no padding and stride is 1, a 5×5 image with a 3×3 kernel produces a 3×3 feature map. The general output-size calculation is (N-F)/S + 1 for valid convolutio'''

# Question 1: Manual Convolution

image = [
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
]

kernel = [
    [-1, -1, -1],
    [0, 0, 0],
    [1, 1, 1]
]

feature_map = []

# Move kernel over image
for i in range(3):
    row = []

    for j in range(3):
        total = 0

        # Multiplication and addition
        for ki in range(3):
            for kj in range(3):
                total = total + image[i + ki][j + kj] * kernel[ki][kj]

        row.append(total)

    feature_map.append(row)

print("Feature Map:")
for row in feature_map:
    print(row)