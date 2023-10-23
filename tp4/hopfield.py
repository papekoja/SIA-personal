import numpy as np

class HopfieldNetwork:
    def __init__(self, size):
        self.weights = np.zeros((size, size))
        
    def train(self, patterns):
        for pattern in patterns:
            # Convert the pattern into a vector
            pattern = pattern.flatten()
            # Outer product update rule
            self.weights += np.outer(pattern, pattern)
        # Remove self-connections
        np.fill_diagonal(self.weights, 0)
        
    def recall(self, pattern, max_iterations=10):
        # Convert the pattern into a vector
        pattern = pattern.flatten()
        for _ in range(max_iterations):
            for i in range(len(pattern)):
                # Compute new state for the ith neuron
                pattern[i] = 1 if np.dot(self.weights[i], pattern) > 0 else -1
        return pattern.reshape((5, 5))
    
# Function that adds noise to a pattern
def add_noise(pattern, noise_level=0.2):
    noisy_pattern = np.copy(pattern)
    num_corrupted = int(noise_level * len(pattern))
    # Randomly choose indices to flip
    indices = np.random.choice(len(pattern), num_corrupted)
    for index in indices:
        noisy_pattern[index] *= -1
    return noisy_pattern

# Create a sample 5x5 pattern
pattern1 = np.array([
    [1, 1,  1, 1, 1],
    [-1, -1, -1, 1, -1],
    [-1, -1, -1, 1, -1],
    [1, -1,  -1, 1, -1],
    [1, 1, 1, -1, -1]
])

# Initialize and train the network
network = HopfieldNetwork(5*5)
network.train([pattern1])

# Generate noisy picture
noisy_pattern = add_noise(pattern1, noise_level=0.2)

reconstructed_pattern = network.recall(noisy_pattern)
print(reconstructed_pattern)