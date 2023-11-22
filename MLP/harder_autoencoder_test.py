import numpy as np
from autoencoder import Autoencoder
import matplotlib.pyplot as plt

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
font_data = read_font_data('MLP/font.h')
X = convert_dataset(font_data)

# Define the size of the input and latent layer
input_size = len(X[0])  # Size of the input data
hidden_size = input_size        # Size of the compressed representation
latent_size = input_size        # Size of the compressed representation

# Initialize the Autoencoder
autoencoder = Autoencoder(input_size=input_size, hidden_size=hidden_size, latent_size=latent_size)

# Train the Autoencoder on only the first pattern
autoencoder.train(X[0], epochs=200)

# Reconstruct data using the trained Autoencoder
reconstructed_X = autoencoder.reconstruct(X[0])

# Reshape the first pattern and its reconstruction into a 2D grid
# Adjust the reshape dimensions as per the actual dimensions of your font patterns
original_pattern = np.reshape(X[0], (7, 5))  # Example: reshape into a 7x5 grid
reconstructed_pattern = np.reshape(reconstructed_X, (7, 5))

round = lambda x: 1 if x >= 0.5 else 0
round_vector = np.vectorize(round)
reconstructed_pattern = round_vector(reconstructed_pattern)

# Create a figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Plot the original pattern
axes[0].imshow(original_pattern, cmap='gray')
axes[0].set_title('Original Pattern')
axes[0].axis('off')  # Turn off axis numbers and ticks

# Plot the reconstructed pattern
axes[1].imshow(reconstructed_pattern, cmap='gray')
axes[1].set_title('Reconstructed Pattern')
axes[1].axis('off')

# Display the plot
plt.show()