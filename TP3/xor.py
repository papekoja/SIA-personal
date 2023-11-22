import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from nn import NeuralNetwork
from sklearn.metrics import confusion_matrix

# Training parameters
architecture = [3, 1]
input_dim = 2
lr = 0.2
epochs = 5000
error_threshold = 0.1

# Training data
x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([[0], [1], [1], [0]])

# Initialize Neural Network
nn = NeuralNetwork(architecture, input_dim)

# List to store errors at each epoch
errors_per_epoch = []

# Training loop
for epoch in range(epochs):
    total_error = 0
    for x, y in zip(x_train, y_train):
        output = nn.train(x, y, lr)
        total_error += nn.calculate_error(output, y)

    # Store the error for this epoch
    errors_per_epoch.append(total_error)

    if total_error < error_threshold:
        break

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Error: {total_error}")

# Plotting the error over epochs
plt.figure(figsize=(10, 6))
plt.plot(errors_per_epoch, label='Training Error')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.title('Error Rate over Epochs')
plt.legend()
plt.show()

# Generate a grid of points over the input space
x_min, x_max = -0.5, 1.5
y_min, y_max = -0.5, 1.5
resolution = 0.01
xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                     np.arange(y_min, y_max, resolution))

# Flatten the grid so the values in each axis are in a single array
grid = np.c_[xx.ravel(), yy.ravel()]

# Predict the function output for each input in the grid
Z = np.array([nn.forward_propagation(i) for i in grid])
Z = Z.reshape(xx.shape)

# Plot the contour and training examples
plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train[:, 0], s=40, edgecolor='k')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("Decision Boundary")
plt.xlabel("Input 1")
plt.ylabel("Input 2")
plt.show()

# Learning rates to test
learning_rates = np.arange(0.01, 0.31, 0.01)  # From 0.1 to 1 in steps of 0.1

# Store errors for each learning rate
errors_for_learning_rates = {lr: [] for lr in learning_rates}

# Training loop for each learning rate
for lr in learning_rates:
    nn = NeuralNetwork(architecture, input_dim)
    for epoch in range(epochs):
        total_error = 0
        for x, y in zip(x_train, y_train):
            output = nn.train(x, y, lr)
            total_error += nn.calculate_error(output, y)

        errors_for_learning_rates[lr].append(total_error)

        if total_error < error_threshold:
            break

# Plotting the error over epochs for each learning rate
plt.figure(figsize=(10, 6))
for lr, errors in errors_for_learning_rates.items():
    plt.plot(errors, label=f'LR={lr}')

plt.xlabel('Epoch')
plt.ylabel('Error')
plt.title('Error Rate over Epochs for Different Learning Rates')
plt.legend()
plt.show()