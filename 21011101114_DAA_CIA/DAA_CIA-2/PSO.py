import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the digits dataset and split it into training and testing sets
digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.3, random_state=42)

# One-hot encode the target variable
encoder = OneHotEncoder(categories='auto')
y_train = encoder.fit_transform(y_train.reshape(-1, 1)).toarray()

# Define the neural network architecture
input_size = X_train.shape[1]
hidden_size = 32
output_size = y_train.shape[1]

# Define the PSO parameters
num_particles = 20
num_iterations = 100
c1 = 2.0
c2 = 2.0
w = 0.9
min_bound = -1.0
max_bound = 1.0

# Define the sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the neural network forward pass
def forward_pass(X, weights):
    W1, W2 = weights
    hidden_layer = sigmoid(np.dot(X, W1))
    output_layer = sigmoid(np.dot(hidden_layer, W2))
    return output_layer

# Define the fitness function to evaluate the solutions
def fitness_function(weights):
    predictions = forward_pass(X_train, weights)
    accuracy = accuracy_score(np.argmax(y_train, axis=1), np.argmax(predictions, axis=1))
    return accuracy

# Initialize the particle positions and velocities
particle_positions = np.random.uniform(min_bound, max_bound, size=(num_particles, input_size * hidden_size + hidden_size * output_size))
particle_velocities = np.zeros((num_particles, input_size * hidden_size + hidden_size * output_size))
particle_best_positions = particle_positions.copy()
particle_best_fitness = np.zeros(num_particles)

# Evaluate the initial particle positions
for i in range(num_particles):
    particle_weights = (particle_positions[i, :input_size*hidden_size].reshape(input_size, hidden_size),
                        particle_positions[i, input_size*hidden_size:].reshape(hidden_size, output_size))
    particle_fitness = fitness_function(particle_weights)
    particle_best_fitness[i] = particle_fitness

# Update the global best solution
global_best_position = particle_positions[np.argmax(particle_best_fitness)]
global_best_fitness = np.max(particle_best_fitness)

# Implement the PSO algorithm
for i in range(num_iterations):
    # Update the particle velocities and positions
    for j in range(num_particles):
        # Compute the new velocity
        r1 = np.random.uniform(size=(input_size * hidden_size + hidden_size * output_size,))
        r2 = np.random.uniform(size=(input_size * hidden_size + hidden_size * output_size,))
        particle_velocities[j] = w * particle_velocities[j] + c1 * r1 * (particle_best_positions[j] - particle_positions[j]) + c2 * r2 * (global_best_position - particle_positions[j])
        
        # Clip the velocity to the maximum bounds
        particle_velocities[j] = np
