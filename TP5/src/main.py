import matplotlib.pyplot as plt
import numpy as np
from  src.models import Autoencoder

def read_font_data(file_path):
    font_data = []
    with open(file_path, 'r') as file:
        for line in file:
            # Strip inline comments and leading/trailing whitespace
            line = line.split('//')[0].strip()

            # Check if the remaining part of the line contains font data
            if line.startswith('{') and line.endswith('},'):
                # Extract the hexadecimal numbers
                hex_values = line.strip('{').strip('},').split(',')
                # Convert hex values to binary and store them
                binary_values = [format(int(hx.strip(), 16), '05b') for hx in hex_values]
                font_data.append(binary_values)
    return font_data

# Convert the dataset to a suitable format
def convert_dataset(font_data):
    dataset = []
    for pattern in font_data:
        flattened_pattern = [int(bit) for row in pattern for bit in row]
        dataset.append(flattened_pattern)
    return np.array(dataset)

# Read and convert the font data
font_data = read_font_data('TP5/font.h')
X = convert_dataset(font_data)

# Define the size of the input and latent layer
input_size = len(X[0])  # Size of the input data
hidden_size = 15        # Size of the compressed representation

# Initialize the Autoencoder
autoencoder = Autoencoder(input_size=input_size, hidden_size=hidden_size)

# Train the Autoencoder
autoencoder.train(X, epochs=200, learning_rate=0.1)

# Reconstruct data using the trained Autoencoder
reconstructed_X = autoencoder.reconstruct(X)


def plot_images(original, reconstructed, num_images=10):
    plt.figure(figsize=(20, 4))
    for i in range(num_images):
        # Display original
        ax = plt.subplot(2, num_images, i + 1)
        plt.imshow(original[i].reshape(7, 5))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstruction
        ax = plt.subplot(2, num_images, i + 1 + num_images)
        plt.imshow(reconstructed[i].reshape(7, 5))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

plot_images(X, reconstructed_X)