import numpy as np
from perceptron_mul_2 import perceptron_mul_2
import matplotlib.pyplot as plt  

# Crear una instancia de la clase perceptron_mul_2 con 2 neuronas en la capa oculta
perceptron = perceptron_mul_2()
perceptron.W1 = np.random.rand(2, 2) - 0.5
perceptron.B1 = np.random.rand(2, 1) - 0.5
perceptron.W2 = np.random.rand(2, 2) - 0.5
perceptron.B2 = np.random.rand(2, 1) - 0.5

# Datos de entrada y etiquetas para la función XOR
X = np.array([[-1, 1, -1, 1], [1, -1, -1, 1]])  # Forma: (2, 4)
Y = np.array([1, 1, -1, -1])

# Hiperparámetros
iterations = 100
learning_rate = 0.002

# Listas para almacenar la precisión y pérdida en cada iteración
accuracy_history = []
loss_history = []

# Entrenar el modelo y guardar métricas en cada iteración
for i in range(iterations):
    trained_W1, trained_B1, trained_W2, trained_B2 = perceptron.gradient_descent(X, Y, 1, learning_rate)
    predictions = perceptron.test_predictions(X)
    accuracy = perceptron.getAccuracy(predictions, Y)
    loss = np.mean((predictions - Y) ** 2)  # Pérdida cuadrática media
    accuracy_history.append(accuracy)
    loss_history.append(loss)

# Realizar predicciones en los mismos datos de entrenamiento
predictions = perceptron.test_predictions(X)

# Calcular la precisión de las predicciones final
accuracy = perceptron.getAccuracy(predictions, Y)
print("Precisión en los datos de entrenamiento:", accuracy)

# Crear gráficas de precisión y pérdida durante el entrenamiento
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(range(iterations), accuracy_history)
plt.title('Precisión durante el entrenamiento')
plt.xlabel('Iteraciones')
plt.ylabel('Precisión')

plt.subplot(1, 2, 2)
plt.plot(range(iterations), loss_history)
plt.title('Pérdida durante el entrenamiento')
plt.xlabel('Iteraciones')
plt.ylabel('Pérdida')

plt.tight_layout()
plt.show()

