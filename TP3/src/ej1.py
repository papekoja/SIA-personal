import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from perceptron import Perceptron

# Función para entrenar y probar el perceptrón en un problema lógico
def train_and_test_logical_function(x, y, function_name):
    print(f"Entrenando para la función lógica {function_name}")
    
    # Crear un DataFrame con los datos de entrada y salida esperada
    data = pd.DataFrame({'x1': x[:, 0], 'x2': x[:, 1], 'y': y})
    
    # Crear una instancia del perceptrón
    perceptron = Perceptron()
    
    # Entrenar el perceptrón
    perceptron.train1(data)
    
    # Realizar predicciones
    predictions = perceptron.predict1(data)
    
    # Imprimir resultados
    print("Resultados:")
    print(predictions)
    print("\nPesos finales del perceptrón:")
    print(perceptron.get_weights())

# Definir datos de entrada y salidas esperadas para las funciones lógicas "Y" y "O exclusivo"
x_and = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
y_and = np.array([-1, -1, -1, 1])

x_xor = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
y_xor = np.array([1, 1, -1, -1])

# Entrenar y probar el perceptrón para la función lógica "Y"
train_and_test_logical_function(x_and, y_and, "Y")

# Entrenar y probar el perceptrón para la función lógica "O exclusivo"
train_and_test_logical_function(x_xor, y_xor, "O exclusivo")




# Función para entrenar y probar el perceptrón en un problema lógico y mostrar gráficas

def train_and_test_logical_function_with_plots(x, y, function_name):
    print(f"Entrenando para la función lógica {function_name}")
    
    # Crear un DataFrame con los datos de entrada y salida esperada
    data = pd.DataFrame({'x1': x[:, 0], 'x2': x[:, 1], 'y': y})
    
    # Crear una instancia del perceptrón
    perceptron = Perceptron()
    
    # Listas para almacenar los cambios de pesos, errores y precisiones durante el entrenamiento
    weight_changes = []
    errors = []
    accuracies = []
    
    # Entrenar el perceptrón
    for epoch in range(100):  # Número de épocas de entrenamiento
        perceptron.train1(data)  # Utiliza train1 para entrenar solo con x1 y x2
        weight_changes.append(perceptron.get_weights())
        predictions = perceptron.predict1(data)  # Utiliza predict1 para hacer predicciones con x1 y x2
        errors.append(np.mean(np.abs(data['y'] - predictions['y_pred'])))  # Compara con 'y' en lugar de 'y_pred'
        
        # Calcular la precisión y agregarla a la lista de precisión
        correct_predictions = (data['y'] == predictions['y_pred']).sum()
        accuracy = correct_predictions / len(data)
        accuracies.append(accuracy)
    
    # Crear gráficas
    plt.figure(figsize=(12, 5))
    
    # Gráfica de los cambios en los pesos a lo largo del entrenamiento
    plt.subplot(1, 3, 1)
    weight_changes = np.array(weight_changes)
    plt.plot(weight_changes[:, 0], label='W1')
    plt.plot(weight_changes[:, 1], label='W2')
    plt.plot(weight_changes[:, 2], label='B')
    plt.xlabel('Época')
    plt.ylabel('Peso')
    plt.legend()
    plt.title('Cambios en los pesos durante el entrenamiento')
    
    # Gráfica del error a lo largo del entrenamiento
    plt.subplot(1, 3, 2)
    plt.plot(errors)
    plt.xlabel('Época')
    plt.ylabel('Error')
    plt.title('Error durante el entrenamiento')
    
    # Gráfica de la precisión a lo largo del entrenamiento
    plt.subplot(1, 3, 3)
    plt.plot(accuracies)
    plt.xlabel('Época')
    plt.ylabel('Precisión')
    plt.title('Precisión durante el entrenamiento')
    
    plt.tight_layout()
    plt.show()


    plt.figure(figsize=(6, 6))
    plt.scatter(data['x1'], data['x2'], c=data['y'], cmap='coolwarm')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title(f'Clasificación final para {function_name}')
    plt.show()


    
    plt.figure(figsize=(6, 6))
    x1_range = np.linspace(-1.5, 1.5, 100)
    x2_range = np.linspace(-1.5, 1.5, 100)
    xx, yy = np.meshgrid(x1_range, x2_range)
    grid_data = pd.DataFrame({'x1': xx.ravel(), 'x2': yy.ravel()})
    predictions = perceptron.predict1(grid_data)
    Z = predictions['y_pred'].values.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap='coolwarm', alpha=0.5)
    plt.scatter(data['x1'], data['x2'], c=data['y'], cmap='coolwarm')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title(f'Función de Decisión Final para {function_name}')
    plt.show()

# Entrenar y probar el perceptrón para la función lógica "Y" con gráficas
train_and_test_logical_function_with_plots(x_and, y_and, "Y")

# Entrenar y probar el perceptrón para la función lógica "O exclusivo" con gráficas
train_and_test_logical_function_with_plots(x_xor, y_xor, "O exclusivo")
