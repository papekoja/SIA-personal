import numpy as np
from multilayer_perceptron import MLP

# Read the input data from the text file
with open('TP5/TP3-ej3-digitos.txt', 'r') as file:
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
    return X, Y  # Retorna X (imágenes) y Y (etiquetas)


# Add noise to the data
def noise_data(X, precentage):
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if np.random.rand() < precentage:
                X[i][j] = 1 if X[i][j] == 0 else 0
    return X

# Define the architecture and activation functions for each layer
layer_sizes = [35, 10, 10]  # Example: input, hidden1, hidden2, output

def test_mlp(mlp, X_test, Y_test):
    """
    Test the MLP model.

    :param mlp: The trained MLP model.
    :param X_test: Test data (features).
    :param Y_test: True labels for the test data.
    :return: Performance metric (e.g., accuracy).
    """
    # Making predictions
    predictions = mlp.predict(X_test)

    # Convert predictions to labels (e.g., using argmax if your output is one-hot encoded)
    predicted_labels = np.argmax(predictions, axis=1)

    print("Predicted labels:", predicted_labels)

    # Convert true labels to one-hot encoded labels
    Y_test = np.eye(10)[Y_test]

    # Calculate accuracy or other metrics
    accuracy = np.mean(predicted_labels == Y_test)

    return accuracy

mlp = MLP(input_size=35, hidden_layers=[10, 10], output_size=10)

# generate training data
X_train, Y_train = generate_data(1000, 0.0)

# Train the MLP
mlp.train(X_train, Y_train, epochs=100, learning_rate=0.1)

# generate test data
X_test, Y_test = generate_data(10, 0.5)

# Example usage
accuracy = test_mlp(mlp, X_test, Y_test)
print("Test Accuracy:", accuracy)
