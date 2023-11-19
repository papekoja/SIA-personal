import numpy as np

class MLP:
    def __init__(self, input_size, hidden_layers, output_size):
        # Inicialización de parámetros
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size

        # Inicialización de pesos y sesgos para capas ocultas y de salida
        self.weights = []
        self.biases = []

        # Inicialización de capas ocultas
        layer_sizes = [input_size] + hidden_layers + [output_size]
        for i in range(len(layer_sizes) - 1):
            weight_matrix = np.random.randn(layer_sizes[i], layer_sizes[i+1])
            bias_vector = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, x):
        # Propagación hacia adelante
        self.layer_outputs = []
        self.layer_inputs = []

        layer_output = x
        for i in range(len(self.weights)):
            layer_input = np.dot(layer_output, self.weights[i]) + self.biases[i]
            layer_output = self.sigmoid(layer_input)
            self.layer_outputs.append(layer_output)
            self.layer_inputs.append(layer_input)

        return layer_output

    def backward(self, x, y, learning_rate):
        # Retropropagación del error y actualización de pesos y sesgos
        output_error = y - self.layer_outputs[-1]
        delta_output = output_error * self.sigmoid_derivative(self.layer_outputs[-1])

        for i in range(len(self.weights) - 1, -1, -1):
            output_delta = delta_output
            delta_output = output_delta.dot(self.weights[i].T) * self.sigmoid_derivative(self.layer_outputs[i])

            self.weights[i] += self.layer_outputs[i - 1].T.dot(output_delta) * learning_rate
            self.biases[i] += np.sum(output_delta, axis=0, keepdims=True) * learning_rate

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            for i in range(len(X)):
                x = X[i]
                target = y[i]
                output = self.forward(x)
                self.backward(x, target, learning_rate)

    def predict(self, X):
        predictions = []
        for x in X:
            output = self.forward(x)
            predictions.append(output)
        return np.array(predictions)

# Ejemplo de uso
if __name__ == "__main__":
    # Datos de entrenamiento
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # Crear y entrenar un MLP
    mlp = MLP(input_size=2, hidden_layers=[2], output_size=1)
    mlp.train(X, y, epochs=10000, learning_rate=0.1)

    # Realizar predicciones
    predictions = mlp.predict(X)
    print("Predicciones:")
    print(predictions)
