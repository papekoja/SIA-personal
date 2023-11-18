import numpy as np
from multilayer_perceptron import MLP

class Autoencoder:
    def __init__(self, input_size, hidden_layers, latent_size):
        # Encoder initialization
        self.encoder = MLP(input_size, hidden_layers, latent_size)

        # Decoder initialization - it's a mirror of the encoder
        reversed_layers = hidden_layers[::-1]
        self.decoder = MLP(latent_size, reversed_layers, input_size)

    def forward(self, x):
        # Encoding
        encoded = self.encoder.forward(x)

        # Decoding
        reconstructed = self.decoder.forward(encoded)
        return reconstructed

    def backward(self, x, learning_rate):
        # Forward pass
        encoded = self.encoder.forward(x)
        reconstructed = self.decoder.forward(encoded)

        # Calculate reconstruction error
        reconstruction_error = x - reconstructed

        # Backward pass for decoder
        self.decoder.backward(encoded, reconstruction_error, learning_rate)

        # Backward pass for encoder
        self.encoder.backward(x, self.decoder.layer_outputs[0], learning_rate)

    def train(self, X, epochs, learning_rate):
        for epoch in range(epochs):
            for x in X:
                self.forward(x)
                self.backward(x, learning_rate)

    def reconstruct(self, X):
        # Reconstructing the input data
        reconstructions = []
        for x in X:
            reconstructions.append(self.forward(x))
        return np.array(reconstructions)

""" # Example usage
if __name__ == "__main__":
    # Example data - could be anything that suits your problem
    X = np.array([...])  # Fill with your data

    # Create and train an Autoencoder
    autoencoder = Autoencoder(input_size=..., hidden_layers=[...], latent_size=...)
    autoencoder.train(X, epochs=10000, learning_rate=0.1)

    # Reconstruct data
    reconstructed_X = autoencoder.reconstruct(X)
    print("Reconstructed Data:")
    print(reconstructed_X)
 """