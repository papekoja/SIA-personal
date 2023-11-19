import numpy as np
from perceptron_mul import MultilayerPerceptron 
import matplotlib.pyplot as plt

def run():
    # Definir las entradas (X) y las salidas esperadas (y) para la función XOR
    X = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
    y = np.array([1, 1, -1, -1])

    # Definir hiperparámetros
    input_size = 2
    hidden_size = 4
    output_size = 1
    learning_rate = 0.1
    epochs = 10000

    # Crear una instancia del perceptrón con Gradiente Descendente
    mlp_gradient_descent = MultilayerPerceptron(input_size, hidden_size, output_size, learning_rate)

    # Crear listas para almacenar el error en cada época
    error_gradient_descent = []

    # Entrenar con Gradiente Descendente y realizar un seguimiento del error
    for epoch in range(epochs):
        error = mlp_gradient_descent.train_with_gradient_descent(X, y.reshape(-1, 1), 1)  # Entrenar una época
        error_gradient_descent.append(error)

    # Crear una instancia del perceptrón con el optimizador Adam
    mlp_adam = MultilayerPerceptron(input_size, hidden_size, output_size, learning_rate)

    # Crear listas para almacenar el error en cada época
    error_adam = []

    # Entrenar con el optimizador Adam y realizar un seguimiento del error
    for epoch in range(epochs):
        error = mlp_adam.train_with_adam(X, y.reshape(-1, 1), 1)  # Entrenar una época
        error_adam.append(error)

    # Crear una gráfica para mostrar cómo cambia el error durante el entrenamiento
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), error_gradient_descent, label="Error (Gradiente Descendente)")
    plt.plot(range(1, epochs + 1), error_adam, label="Error (Adam)")
    plt.xlabel("Épocas")
    plt.ylabel("Error")
    plt.legend()
    plt.title("Entrenamiento de Perceptrón Multicapa")
    plt.grid(True)
    plt.show()

    # Probar el modelo entrenado con XOR
    test_input = np.array([[1, -1]])
    output_gradient_descent = mlp_gradient_descent.predict(test_input)
    output_adam = mlp_adam.predict(test_input)

    print("Resultado con Gradiente Descendente:", output_gradient_descent)
    print("Resultado con Adam:", output_adam)

run()
