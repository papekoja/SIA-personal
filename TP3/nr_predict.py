import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from nn import NeuralNetwork
from sklearn.metrics import confusion_matrix

def generate_data(qty, noise_percentage):
    X = []
    Y = []
    for i in range(qty):
        n = np.random.randint(0, 10)
        pic = noise_data(numbers[n], noise_percentage)
        X.append(pic.flatten())
        Y.append(n)  # Actual digit label
    X = np.array(X)
    Y = np.array(Y)
    return X.T, Y

def noise_data(X, percentage):
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if np.random.rand() < percentage:
                X[i][j] = 1 if X[i][j] == 0 else 0
    return X

# Read the input data from the text file
with open('TP3/TP3-ej3-digitos.txt', 'r') as file:
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

# Create training and testing datasets
X_train, Y_train = generate_data(1000, 0)
X_test, Y_test = generate_data(1000, 0)

def one_hot_encode(y, num_classes=10):
    return np.eye(num_classes)[y]

Y_train_encoded = one_hot_encode(Y_train)
Y_test_encoded = one_hot_encode(Y_test)


# Neural Network parameters
input_dim = 35  # 5x7 grid flattened
architecture = [64, 10]  # Adjusted architecture
lr = 0.1
epochs = 1000

# Initialize Neural Network
nn = NeuralNetwork(architecture, input_dim)

errors_per_epoch = []

for epoch in range(epochs):
    total_loss = 0
    for i in range(X_train.shape[1]):
        x = X_train[:, i]
        y = Y_train_encoded[:, i]
        output = nn.train(x, y, lr)
        total_loss += nn.calculate_error(output, y)
    
    errors_per_epoch.append(total_loss)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Total Loss: {total_loss}")


# Testing the model
correct_predictions = 0
for i in range(X_test.shape[1]):
    x = X_test[:, i]
    predicted_output = nn.forward_propagation(x)
    predicted_class = np.argmax(predicted_output)
    actual_class = np.argmax(Y_test_encoded[:, i])
    if predicted_class == actual_class:
        correct_predictions += 1

accuracy = correct_predictions / X_test.shape[1]
print(f"Test Accuracy: {accuracy}")

# Plotting the error over epochs
plt.figure(figsize=(10, 6))
plt.plot(errors_per_epoch, label='Training Error')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.title('Error Rate over Epochs')
plt.legend()
plt.show()

# Generate predictions for test data
test_predictions = []
for i in range(X_test.shape[1]):
    x = X_test[:, i]
    predicted_output = nn.forward_propagation(x)
    test_predictions.append(np.round(predicted_output))

test_predictions = np.array(test_predictions).flatten()

# Calculate the confusion matrix
cm = confusion_matrix(Y_test, test_predictions)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()