import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

digits = pd.read_csv('digits_test.csv', header=None).values
labels = pd.read_csv('digits_keys.csv', header=None).values.flatten()

# Normalize pixel values to the range [0, 1].
digits = digits / 255.0

# SOM parameters:
som_size = 10  # Size of the SOM grid (10x10).
input_len = digits.shape[1]  # Length of the input vectors (784 for 28x28 images).
initial_learning_rate = 0.2  # Initial learning rate.
initial_sigma = max(som_size, som_size) / 2  # Initial neighborhood radius.
num_epochs = 20

# Initialize the weight vectors with small random values around the mean of the input vectors.
weights = np.random.rand(som_size, som_size, input_len) * 0.01 + np.mean(digits, axis=0)


# Hexagonal grid directions.
hex_directions = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]


#  convert from axial coordinates (q, r) to cube coordinates (x, y, z).
def axial_to_cube(axial):
    q, r = axial
    return (q, r, -q - r)


# Calculates the cube distance between two points.
def cube_distance(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]), abs(a[2] - b[2]))


# Finds the best matching uint (a.k.a BMU) for a given input vector.
def find_bmu(input_vec):
    # Calculates the Euclidean distance between the input vector and each weight vector
    # and stores the index of the closest one.
    bmu_idx = np.argmin(np.sum((weights - input_vec) ** 2, axis=2))

    # Convert the flat index to 2D index (row, col) in the SOM grid.
    bmu = np.unravel_index(bmu_idx, (som_size, som_size))
    return bmu


# Find the second best matching unit (SBMU) for a given input vector.
def find_sbmu(input_vec, bmu):
    distances = np.sum((weights - input_vec) ** 2, axis=2)
    distances[bmu[0], bmu[1]] = np.inf  # Exclude the BMU.
    sbmu_idx = np.argmin(distances)
    sbmu = np.unravel_index(sbmu_idx, (som_size, som_size))
    return sbmu


# Updates the weights of the SOM.
def update_weights(input_vec, bmu, iteration, total_iterations):
    learning_rate = initial_learning_rate * np.exp(-iteration / (total_iterations / 5))
    if iteration < total_iterations / 3:
        sigma = initial_sigma * np.exp(-iteration / (total_iterations / 2.5))
    else:
        sigma = initial_sigma * np.exp(-iteration / (total_iterations / 2.5))

    bmu_cube = axial_to_cube(bmu)
    for i in range(som_size):
        for j in range(som_size):
            neuron = np.array([i, j])
            neuron_cube = axial_to_cube(neuron)
            distance = cube_distance(bmu_cube, neuron_cube)
            if distance < sigma:
                influence = np.exp(-distance ** 2 / (2 * (sigma ** 2)))
                weights[i, j] += influence * learning_rate * (input_vec - weights[i, j])


# Training the SOM:
total_iterations = num_epochs * digits.shape[0]
iteration = 0
start_time = time.time()

for epoch in range(num_epochs):
    indices = np.arange(digits.shape[0])
    np.random.shuffle(indices)
    for idx in indices:
        input_vec = digits[idx]
        bmu = find_bmu(input_vec)
        update_weights(input_vec, bmu, iteration, total_iterations)
        iteration += 1

    elapsed_time = time.time() - start_time
    if elapsed_time > 180:  # Stop if running time exceeds 3 minutes.
        print("timeout!!!")
        break

# Map each input vector to its BMU and SBMU:
bmu_map = np.array([find_bmu(x) for x in digits])
sbmu_map = np.array([find_sbmu(x, bmu) for x, bmu in zip(digits, bmu_map)])

# Calculate the distances between BMU and SBMU
distances = np.array([cube_distance(axial_to_cube(bmu), axial_to_cube(sbmu)) for bmu, sbmu in zip(bmu_map, sbmu_map)])

# Evaluate the goodness of the output
goodness = np.mean(distances <= 1)

# Quantization Error:
quantization_error = np.mean(
    [np.linalg.norm(digits[idx] - weights[bmu_map[idx][0], bmu_map[idx][1]]) for idx in range(len(digits))])
print(f'Quantization Error: {quantization_error:.4f}')

# Topographic Error:
topographic_error = np.mean(
    [cube_distance(axial_to_cube(bmu), axial_to_cube(sbmu)) > 1 for bmu, sbmu in zip(bmu_map, sbmu_map)])
print(f'Topographic Error: {topographic_error:.4f}')

# Calculate dominant digit and its frequency for each neuron
dominant_digit = np.zeros((som_size, som_size), dtype=int)
dominant_digit_freq = np.zeros((som_size, som_size), dtype=float)

for i in range(som_size):
    for j in range(som_size):
        # Get the labels of all input vectors mapped to the neuron (i, j).
        bmu_labels = labels[(bmu_map[:, 0] == i) & (bmu_map[:, 1] == j)]
        if len(bmu_labels) > 0:
            # Find the most frequent label (dominant digit) and its frequency.
            unique, counts = np.unique(bmu_labels, return_counts=True)
            dominant_digit[i, j] = unique[np.argmax(counts)]
            dominant_digit_freq[i, j] = np.max(counts) / len(bmu_labels)

# Visualization of dominant digits and frequencies:
plt.figure(figsize=(10, 10))
for i in range(som_size):
    for j in range(som_size):
        plt.text(j, i, f'{dominant_digit[i, j]}\n({dominant_digit_freq[i, j] * 100:.1f}%)',
                 ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5))
plt.xticks(range(som_size))
plt.yticks(range(som_size))
plt.gca().invert_yaxis()
plt.grid()
plt.title(f'Dominant Digit and Frequency for Each Neuron')
plt.show()

# Visualization of the weight vectors:
plt.figure(figsize=(10, 10))
for i in range(som_size):
    for j in range(som_size):
        plt.subplot(som_size, som_size, i * som_size + j + 1)
        plt.imshow(weights[i, j].reshape(28, 28), cmap='gray')
        plt.axis('off')
plt.suptitle(
    f'Weight Vectors of Neurons\n'
    f'Quantization Error: {quantization_error:.2f}\n'
    f'topographic_error: {topographic_error:.2f}')
plt.show()
