import numpy as np
import pandas as pd
from nn import MultilayerPerceptron

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
    #change Y to one hot encoding
    Y = pd.get_dummies(Y).values
    return X, Y

# Add noise to the data
def noise_data(X, precentage):
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if np.random.rand() < precentage:
                X[i][j] = 1 if X[i][j] == 0 else 0
    return X

p = MultilayerPerceptron(35, 15, 10)
X_train, Y_train = generate_data(100, 0.01)
X_test, Y_test = generate_data(1000, 0.01)

p.train(X_train, Y_train, 10000, 0.1)
predictions = p.predict(X_test)
predictions = np.round(predictions)

# print accuracy
print("Accuracy: ", np.sum(predictions == Y_test) / Y_test.shape[0])

# convert one hot encoding to number
Y_test = np.argmax(Y_test, axis=1)
predictions = np.argmax(predictions, axis=1)

# Print 10 first predictions and real values
print("Predictions: ", predictions[:10], " Real values: ", Y_test[:10])