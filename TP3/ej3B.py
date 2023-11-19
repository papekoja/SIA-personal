import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read the input data from the text file
with open('sia-tp3/data/TP3-ej3-digitos.txt', 'r') as file:
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

class perceptron_mul_2:
    def __init__(self):
        # Create weights for the first layer (input is 0th). 10 neurons, 35 inputs(5x7)
        self.W1 = np.random.rand(10, 35) - 0.5
        # Create bias for first layer
        self.B1 = np.random.rand(10, 1) - 0.5
        # Create weights for the second layer (input is 1st). 10 neurons, 10 inputs
        self.W2 = np.random.rand(10, 10) - 0.5
        # Create bias for second layer
        self.B2 = np.random.rand(10, 1) - 0.5
        # Create a variable to control logging
        self.log = False

    #relu activation function for the hidden layer
    def relu(self, Z):
        return np.maximum(0, Z)
    
    #softmax activation function for the output layer
    def softmax(self, Z):
        return np.exp(Z) / np.sum(np.exp(Z), axis=0)

    # Forward propagation
    def forward_prop(self, X):

        X = X.T
        # First layer
        Z1 = self.W1.dot(X) + self.B1
        A1 = self.relu(Z1)
        # Second layer
        Z2 = self.W2.dot(A1) + self.B2
        A2 = self.softmax(Z2)
        return Z1, A1, Z2, A2
    
    # One hot encoding of the output labels (Y). There are 10 classes (Y.max()+1, 0-9)
    def one_hot(self, Y):
        one_hot_Y = np.zeros((Y.size, Y.max()+1))
        one_hot_Y[np.arange(Y.size), Y] = 1
        one_hot_Y = one_hot_Y.T
        return one_hot_Y
    
    # Derivative of the relu activation function
    def deriv_relu(self, Z):
        return Z > 0

# Back propagation
    def back_prop(self, Z1, A1, Z2, A2, X, Y):
        m = Y.size
        one_hot_Y = self.one_hot(Y)
        dZ2 = A2 - one_hot_Y
        dW2 = 1/m * dZ2.dot(A1.T)
        db2 = 1 / m * np.sum(dZ2)
        
        # Transpone X para que las dimensiones sean compatibles
        X = X.T
        
        dZ1 = self.W2.T.dot(dZ2)
        dW1 = 1 / m * dZ1.dot(X.T)
        db1 = 1 / m * np.sum(dZ1)
        
        return dW1, db1, dW2, db2

    
    # Update the weights and bias
    def uptade_params(self, dW1, db1, dW2, db2, learning_rate):
        self.W1 -= learning_rate * dW1
        self.B1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.B2 -= learning_rate * db2

    # Get the predictions from the output layer
    def get_predictions(self, A2):
        return np.argmax(A2, 0)
    
    # Get the accuracy of the predictions
    def getAccuracy(self, predictions, Y):
        return np.sum(predictions == Y) / Y.size
    
    # Train the perceptron with gradient descent
    def gradient_descent(self, X, Y, iterations, alpha):
        errors = []       # Lista para almacenar los errores en cada época
        accuracies = []   # Lista para almacenar la precisión en cada época
        
        for i in range(iterations):
            Z1, A1, Z2, A2 = self.forward_prop(X)
            dW1, db1, dW2, db2 = self.back_prop(Z1, A1, Z2, A2, X, Y)
            self.uptade_params(dW1, db1, dW2, db2, alpha)
            
            error = np.mean(np.abs(A2 - self.one_hot(Y)))  # Calcular el error en cada época
            errors.append(error)         # Almacenar el error en esta época
            
            accuracy = self.getAccuracy(self.get_predictions(A2), Y)  # Calcular la precisión en cada época
            accuracies.append(accuracy)  # Almacenar la precisión en esta época
            
            if (i % 50 == 0) and (self.log != False):
                print("Época:", i, "Precisión:", accuracy, "Error:", error)
        
        return errors, accuracies


    
    # Test the perceptron with the test data
    def test_predictions(self, X):
        _, _, _, A2 = self.forward_prop(X)
        return self.get_predictions(A2)


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

p = perceptron_mul_2()
X_train, Y_train = generate_data(1000, 0)
X_test, Y_test = generate_data(1000, 0)

# Method to train the perceptron with different configurations
def run_config(qty_of_data, noise_precentage, iternations, alpha):
    p = perceptron_mul_2()
    X_train, Y_train = generate_data(qty_of_data, noise_precentage)
    X_test, Y_test = generate_data(1000, noise_precentage)
    errors, accuracies = p.gradient_descent(X_train, Y_train , iternations, alpha)
    predictions = p.test_predictions(X_test)
    return errors, accuracies, Y_test, predictions  # Retorna errores, precisión, etiquetas verdaderas y predicciones


accuracies = []
for i in range(20):
    accuracies.append(run_config(500, 0.01 * i, 1000 , 0.38))

print(accuracies)
print(max(accuracies), accuracies.index(max(accuracies)))


predictions = p.test_predictions(X_test)

# Función para determinar si un número es par o impar
def es_par_o_impar(numero):
    if numero % 2 == 0:
        return "Par"
    else:
        return "Impar"

# Convertir las etiquetas verdaderas en una lista de "Par" o "Impar"
Y_test_par_impar = [es_par_o_impar(y) for y in Y_test]

# Convertir las predicciones en una lista de "Par" o "Impar"
predictions_par_impar = [es_par_o_impar(prediction) for prediction in predictions]

# Calcular la precisión
correct_predictions = sum(1 for i in range(len(Y_test)) if Y_test_par_impar[i] == predictions_par_impar[i])
accuracy = correct_predictions / len(Y_test)

# Imprimir la precisión
print("Precisión en la clasificación de par/impar:", accuracy)

# Entrenar el perceptrón y obtener las listas de error y precisión
errors, accuracies = p.gradient_descent(X_train, Y_train, 1000, 0.38)

# Crear los gráficos
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Gráfico de error
ax1.plot(range(len(errors)), errors)
ax1.set_title('Error a lo largo de las épocas')
ax1.set_xlabel('Época')
ax1.set_ylabel('Error')

# Gráfico de precisión
ax2.plot(range(len(accuracies)), accuracies)
ax2.set_title('Precisión a lo largo de las épocas')
ax2.set_xlabel('Época')
ax2.set_ylabel('Precisión')

plt.show()

accuracies = []
errors = []

for i in range(10):  # Itera sobre cada número del 0 al 9
    error_i, accuracy_i, Y_test_i, predictions_i = run_config(500, 0.01 * i, 1000 , 0.38)
    errors.append(error_i)
    accuracies.append(accuracy_i)
    
    print(f"Configuración {i}:")
    print("Precisión:", accuracy_i[-1])
    print("Error:", error_i[-1])
    
    # Crear gráfico para el número i
    plt.figure(figsize=(8, 4))
    plt.plot(range(len(error_i)), error_i, label="Error")
    plt.plot(range(len(accuracy_i)), accuracy_i, label="Precisión")
    plt.title(f"Error y Precisión para el número {i}")
    plt.xlabel("Época")
    plt.ylabel("Valor")
    plt.legend()
    plt.show()

# Encuentra la configuración con la máxima precisión
max_accuracy = max([accuracy[-1] for accuracy in accuracies])
best_configuration = accuracies.index(max_accuracy)
print(f"La mejor configuración es {best_configuration} con una precisión de {max_accuracy}")



