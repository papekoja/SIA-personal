from multilayer_perceptron import MultilayerPerceptron

class Autoencoder:
    def __init__(self, input_size, hidden_size):
        # The encoder compresses the input
        self.encoder = MultilayerPerceptron(input_size, hidden_size, hidden_size)
        # The decoder tries to reconstruct the input
        self.decoder = MultilayerPerceptron(hidden_size, hidden_size, input_size)

    def forward(self, inputs):
        # Encoder forward pass
        encoded = self.encoder.forward_propagation(inputs)
        # Decoder forward pass
        decoded = self.decoder.forward_propagation(encoded)
        return encoded, decoded

    def train(self, inputs, epochs, learning_rate):
        for epoch in range(epochs):
            # Forward pass
            encoded, decoded = self.forward(inputs)

            # Train the decoder (target is the original inputs)
            self.decoder.train(encoded, inputs, 1, learning_rate)

            # Train the encoder (target is the encoder's output)
            self.encoder.train(inputs, self.decoder.hidden_layer_output, 1, learning_rate)

    def reconstruct(self, inputs):
        # Reconstruct the input from the encoder and decoder
        _, decoded = self.forward(inputs)
        return decoded
