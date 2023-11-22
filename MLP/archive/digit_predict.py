import numpy as np
import pandas as pd
from MLP import MLP

# Read the input data from the text file
with open('MLP/TP3-ej3-digitos.txt', 'r') as file:
    lines = file.readlines()

# Parse the lines and create a list of lists
data = []
for line in lines:
    row = [int(x) for x in line.strip().split()]
    data.append(row)

# Convert the list of lists into a NumPy array
data = np.array(data)

# 7 rows is one image. Split the input array into 7 rows each
numbers = np.split(data, 10)

# Convert the list of lists into a NumPy array. Each element is a 5x7 matrix where corresponding to a digit
numbers = np.array(numbers)

# Generate data with noise percentage
def generate_data(qty, noise_precentage):
    X = []
    Y = []
    for i in range(qty):
        n = np.random.randint(0, 10)
        pic = noise_data(numbers[n], noise_precentage)
        X.append(pic.flatten())
        Y.append(n)
    X = np.array(X)
    Y = np.array(Y)
    return X.T, Y

# Add noise to the data
def noise_data(X, precentage):
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if np.random.rand() < precentage:
                X[i][j] = 1 if X[i][j] == 0 else 0
    return X

p = MLP(35, 10, 10, 0.1)
X_train, Y_train = generate_data(100, 0.01)
X_test, Y_test = generate_data(1000, 0.01)

p.gradient_descent(X_train, Y_train, 1000, 0.1)
predictions = p.test_predictions(X_test)
print("Accuracy: ", p.getAccuracy(predictions, Y_test))
