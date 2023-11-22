import numpy as np
from autoencoder import Autoencoder

# Simple test
if __name__ == "__main__":
    # Initialize autoencoder
    autoencoder = Autoencoder(3, 2, 2)

    # Train autoencoder
    inputs = np.array([[0, 0, 1],
                       [0, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])
    autoencoder.train(inputs, 10000)

    # Test autoencoder
    test_inputs = np.array([[0, 0, 1],
                            [0, 1, 1],
                            [1, 0, 1],
                            [1, 1, 1]])
    latent_layer_output, outputs = autoencoder.forward_propagation(test_inputs)
    print("Input: \n", test_inputs)
    print("Output: \n", outputs)
