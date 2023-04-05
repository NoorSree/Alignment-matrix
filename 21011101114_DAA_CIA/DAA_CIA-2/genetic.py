import torch
import random

# Define the fitness function
def fitness(chromosome):
    # Calculate fitness as the sum of the chromosome
    return torch.sum(chromosome)

# Generate an initial population
def generate_population(population_size, chromosome_length):
    population = []
    for i in range(population_size):
        chromosome = torch.randint(0, 2, size=(chromosome_length,))
        population.append(chromosome)
    return population

# Select parents for mating
def select_parents(population, fitness_scores):
    # Use roulette wheel selection to select parents
    total_fitness = torch.sum(fitness_scores)
    probabilities = fitness_scores / total_fitness
    parent_indices = torch.multinomial(probabilities, 2, replacement=True)
    return [population[i] for i in parent_indices]

# Apply crossover operator
def crossover(parents, crossover_rate):
    if random.random() < crossover_rate:
        crossover_point = random.randint(1, len(parents[0]) - 1)
        child1 = torch.cat([parents[0][:crossover_point], parents[1][crossover_point:]])
        child2 = torch.cat([parents[1][:crossover_point], parents[0][crossover_point:]])
        return [child1, child2]
    else:
        return parents

# Apply mutation operator
def mutate(child, mutation_rate):
    for i in range(len(child)):
        if random.random() < mutation_rate:
            child[i] = 1 - child[i]
    return child

# Main genetic algorithm loop
def genetic_algorithm(population_size, chromosome_length, num_generations, crossover_rate, mutation_rate):
    # Generate initial population
    population = generate_population(population_size, chromosome_length)

    # Loop through generations
    for i in range(num_generations):
        # Evaluate fitness of each chromosome
        fitness_scores = torch.tensor([fitness(chromosome) for chromosome in population])

        # Select parents for mating
        parents = select_parents(population, fitness_scores)

        # Apply crossover operator
        children = crossover(parents, crossover_rate)

        # Apply mutation operator
        mutated_children = [mutate(child, mutation_rate) for child in children]

        # Replace old population with new population
        population = mutated_children

    # Return best solution
    fitness_scores = torch.tensor([fitness(chromosome) for chromosome in population])
    best_index = torch.argmax(fitness_scores)
    return population[best_index], fitness_scores[best_index]

# Example usage
population_size = 100
chromosome_length = 10
num_generations = 1000
crossover_rate = 0.8
mutation_rate = 0.01
best_solution, best_fitness = genetic_algorithm(population_size, chromosome_length, num_generations, crossover_rate, mutation_rate)

print("Best solution: ", best_solution)
print("Best fitness: ", best_fitness)
