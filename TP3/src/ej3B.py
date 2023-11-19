import numpy as np
from sklearn.model_selection import train_test_split
from perceptron_mul import MLP

def cargar_datos(nombre_archivo):
    with open(nombre_archivo, 'r') as archivo:
        lineas = archivo.readlines()

    datos = []
    etiquetas = []

    for i in range(0, len(lineas), 7):
        matriz = [list(map(int, linea.split())) for linea in lineas[i:i+7]]
        vector_caracteristicas = np.array(matriz).flatten()  # Convertir la matriz en un vector unidimensional
        datos.append(vector_caracteristicas)

        # Etiquetar como "par" (0) o "impar" (1) según corresponda
        numero = i // 7
        etiqueta = 0 if numero % 2 == 0 else 1
        etiquetas.append(etiqueta)

    return np.array(datos), np.array(etiquetas)

# Cargar los datos desde el archivo
X, y = cargar_datos("sia-tp3\data\TP3-ej3-digitos.txt")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar la red neuronal
mlp = MLP(input_size=len(X[0]), hidden_layers=[64], output_size=1)
mlp.train(X_train, y_train, epochs=100, learning_rate=0.01)

# Evaluar el rendimiento de la red neuronal en el conjunto de prueba
def evaluar_modelo(modelo, X_test, y_test):
    # Realizar predicciones en el conjunto de prueba
    predictions = modelo.predict(X_test)

    # Convertir las predicciones a etiquetas binarias (0 o 1)
    predictions_binarias = (predictions >= 0.5).astype(int)

    # Calcular la precisión del modelo
    accuracy = np.mean(predictions_binarias == y_test)
    return accuracy

accuracy = evaluar_modelo(mlp, X_test, y_test)
print(f"Precisión en el conjunto de prueba: {accuracy * 100:.2f}%")
