from perceptron_mul import MultilayerPerceptron
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def run():
    # Paso 1: Preparar los datos
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Dividir el conjunto de entrenamiento en entrenamiento y validación
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Paso 2: Entrenar la red con Gradiente Descendente
    input_size = 784
    hidden_size = 128
    output_size = 10
    learning_rate = 0.01

    mlp_gd = MultilayerPerceptron(input_size, hidden_size, output_size, learning_rate)
    
    # Crear listas para registrar la precisión y la pérdida en el conjunto de entrenamiento
    train_loss_history = []
    val_accuracy_history = []

    for epoch in range(100):  # Número de épocas
        train_loss = mlp_gd.train_with_gradient_descent(X_train.reshape(-1, input_size), y_train, epochs=1)
        train_loss_history = mlp_gd.train_with_gradient_descent(X_train.reshape(-1, input_size), y_train, epochs=100)


        # Calcular la precisión en el conjunto de validación en esta época
        val_predictions_gd = mlp_gd.predict(X_val.reshape(-1, input_size))
        y_val_pred_gd = np.argmax(val_predictions_gd, axis=1)
        y_val_true = np.argmax(y_val, axis=1)
        accuracy_gd = accuracy_score(y_val_true, y_val_pred_gd)
        val_accuracy_history.append(accuracy_gd)

        print(f'Época {epoch + 1}/{100}, Pérdida en entrenamiento: {train_loss_history[epoch]:.4f}, Precisión en Validación: {accuracy_gd * 100:.2f}%')

    # Crear gráficos de pérdida en entrenamiento y precisión en validación
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, 101), train_loss_history, label='Pérdida en Entrenamiento', color='blue')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, 101), val_accuracy_history, label='Precisión en Validación', color='green')
    plt.xlabel('Época')
    plt.ylabel('Precisión')
    plt.legend()

    plt.tight_layout()
    plt.show()






    

    # Paso 4: Entrenar la red con Adam
    #mlp_adam = MultilayerPerceptron(input_size, hidden_size, output_size, learning_rate)
    #mlp_adam.train_with_adam(X_train.reshape(-1, input_size), y_train, epochs=100)

    # Paso 5: Evaluar la red entrenada con Adam en el conjunto de validación
    #val_predictions_adam = mlp_adam.predict(X_val.reshape(-1, input_size))
    #y_val_pred_adam = np.argmax(val_predictions_adam, axis=1)
    #accuracy_adam = accuracy_score(y_val_true, y_val_pred_adam)
    #print(f'Precisión en el conjunto de validación con Adam: {accuracy_adam * 100:.2f}%')

    # Paso 6: Comparar los dos métodos de optimización y seleccionar el mejor

    # Opcional: Entrenar y evaluar en el conjunto de prueba el método seleccionado
    test_predictions = None
    selected_method = None

    #if accuracy_gd > accuracy_adam:
    #    selected_method = "Gradiente Descendente"
    #    test_predictions = mlp_gd.predict(X_test.reshape(-1, input_size))
    #else:
    #    selected_method = "Adam"
    #    test_predictions = mlp_adam.predict(X_test.reshape(-1, input_size))

    #y_test_pred = np.argmax(test_predictions, axis=1)
    #y_test_true = np.argmax(y_test, axis=1)
    #accuracy_test = accuracy_score(y_test_true, y_test_pred)
    #print(f'Precisión en el conjunto de prueba con el método seleccionado ({selected_method}): {accuracy_test * 100:.2f}%')

run()