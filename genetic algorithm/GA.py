import numpy as np
import random
import matplotlib.pyplot as plt


# Reads preference data from a file and organize it into preference lists for males and females.
def read_preferences(filename):
    with open(filename, 'r') as file:
        data = file.read().strip().split()

    total_values = len(data)

    n = int((total_values / 2) ** 0.5)

    preferences_females = []
    preferences_males = []

    index = 0  # The index of the current line.
    # Iterating over the n males:
    for i in range(n):
        preferences_males.append([int(x) - 1 for x in data[index:index + n]])
        index += n
    # Iterating over the n females:
    for i in range(n):
        preferences_females.append([int(x) - 1 for x in data[index:index + n]])
        index += n

    return n, preferences_males, preferences_females


# Calculates the normalized fitness score of a matching between males and females.
def fitness(matching, preferences_males, preferences_females):
    fitness = 0

    n = len(matching)
    for male in range(n):
        female = matching[male]
        fitness += (n / 2) - preferences_males[male].index(female)

        fitness += (n / 2) - preferences_females[female].index(male)

    normalized_fitness = (fitness + n ** 2) / (2 * n ** 2)
    return normalized_fitness


# NOT USED!
def fitness_verse2(matching, preferences_males, preferences_females):
    fitness = 0

    n = len(matching)
    for male in range(n):
        female = matching[male]
        fitness += n - preferences_males[male].index(female)

        fitness += n - preferences_females[female].index(male)

    normalized_fitness = fitness / (2 * n)
    return normalized_fitness


# Creates initial population, In our "world" it means population of matching.
def create_initial_population(n, population_size):
    population = []
    for _ in range(population_size):
        matching = list(range(n))
        random.shuffle(matching)
        population.append(matching)

    return population


# NOT USED!
def create_diverse_initial_population(n, population_size):
    population = set()
    while len(population) < population_size:
        matching = list(range(n))
        random.shuffle(matching)
        population.add(tuple(matching))
    return [list(m) for m in population]


# The next 4 functions are related to the Niching process, we implement in case of premature convergence.

# NOT USED!
# Calculates the distance based on the Kendall Tau metric.
def kendall_tau_distance(matching1, matching2):
    n = len(matching1)
    distance = 0
    for i in range(n):
        for j in range(i + 1, n):
            # Check if the pairs (i, j) are in the same relative order in both matchings
            if (matching1[i] < matching1[j] and matching2[i] > matching2[j]) or (
                    matching1[i] > matching1[j] and matching2[i] < matching2[j]):
                distance += 1
    return distance


# Calculates the distance based on the hamming distance metric.
def hamming_distance(matching1, matching2):
    return sum(x != y for x, y in zip(matching1, matching2))


# calculates distances between parts of the population.
def calculate_distances(population, metric):
    n = len(population)
    distances = []
    for i in range(n):
        distances.append([metric(population[i], population[j]) for j in range(n)])
    return distances


# NOT USED!
# adjusts the fitness values of individuals in a population to account for crowding or proximity in the solution space,
# promoting diversity.
def fitness_sharing_verse2(fitness, distances, sharing_threshold):
    adjusted_fitness = []
    n = len(fitness)
    for i in range(n):
        share_sum = sum(1.0 if distances[i][j] < sharing_threshold else 0.0 for j in range(n))
        adjusted_fitness.append(fitness[i] / share_sum)
    return adjusted_fitness


# adjusts the fitness values of individuals in a population to account for crowding or proximity in the solution space,
# promoting diversity.
def fitness_sharing(fitnesses, distances, sharing_threshold):
    size = len(fitnesses)
    shared_fitnesses = np.zeros(size)
    for i in range(size):
        sharing_sum = 1  # Include the individual itself
        for j in range(size):
            if i != j:
                if distances[i][j] < sharing_threshold:
                    sharing_sum += 1 - (distances[i][j] / sharing_threshold) ** 2
        shared_fitnesses[i] = fitnesses[i] / sharing_sum
    return shared_fitnesses


# Performs tournament selection to choose parents based on their fitness. (The selected parents would take part in the
# crossover process in the current iteration).
# More simple explanation would be saying that tournament_size parents chosen randomly
# and the best ranked one are added to returned parents array,
# this process continues until the wanted amount parents are added.
def select_parents_tournament(population, fitnesses, num_parents, tournament_size=7):
    parents = []
    while len(parents) < num_parents:
        tournament = random.sample(list(zip(population, fitnesses)), tournament_size)
        winner = max(tournament, key=lambda x: x[1])[0]
        if winner not in parents:
            parents.append(winner)
    return parents

# NOT USED!
def select_parents_rank_based(population, fitnesses, num_parents):
    # Rank population based on fitness
    sorted_indices = sorted(range(len(fitnesses)), key=lambda k: fitnesses[k], reverse=True)
    rank_probabilities = [(2 * (i + 1)) / (len(fitnesses) * (len(fitnesses) + 1)) for i in range(len(fitnesses))]

    parents = []
    for _ in range(num_parents):
        idx = np.random.choice(len(fitnesses), p=rank_probabilities)
        parents.append(population[idx])
    return parents


# NOT USED!
# Selects a specified number of parents from the population based on their fitness scores.
def select_parents(population, fitnesses, num_parents):
    """The random.choices function in Python is used for weighted random selection.
     This function takes a list of items (the population), corresponding weights (fitness scores),
      and the number of items to select (num_parents)."""
    parents = random.choices(population, weights=fitnesses, k=num_parents)
    return parents


# NOT USED!
def select_parents_roulette(population, fitnesses, num_parents):
    total_fitness = sum(fitnesses)
    selection_probs = [f / total_fitness for f in fitnesses]
    parents = random.choices(population, weights=selection_probs, k=num_parents)
    return parents


# NOT USED!
def select_parents_fitness_proportionate(population, fitnesses, num_parents):
    # Calculate total fitness
    total_fitness = sum(fitnesses)

    # Calculate probabilities for each individual
    probabilities = [fit / total_fitness for fit in fitnesses]

    parents = []
    # for _ in range(num_parents):
    #     # Select a parent based on the calculated probabilities
    #     selected_index = np.random.choice(len(population), p=probabilities)
    #     parents.append(population[selected_index])

    while len(parents) < num_parents:
        selected_index = np.random.choice(len(population), p=probabilities)
        if population[selected_index] not in parents:
            parents.append(population[selected_index])

    return parents


# NOT USED!
# Selects parents from the population using Stochastic Universal Sampling (SUS).
# Ensuring a balanced and effective method for evolving populations towards optimal solutions.
def select_parents_stochastic_universal_sampling(population, fitnesses, num_parents):
    total_fitness = sum(fitnesses)
    pointers = np.random.uniform(0, total_fitness, size=num_parents)
    pointers.sort()

    selected_parents = []
    fitness_sum = 0
    index = 0
    for i in range(len(population)):
        fitness_sum += fitnesses[i]
        while index < num_parents and fitness_sum > pointers[index]:
            selected_parents.append(population[i])
            index += 1

    return selected_parents


# NOT USED!
# Crowding Technique
def crowded_tournament_selection(population, fitnesses, num_parents, crowding_factor=2):
    parents = []
    for _ in range(num_parents):
        candidates = random.sample(list(zip(population, fitnesses)), crowding_factor)
        winner = max(candidates, key=lambda x: x[1])[0]
        parents.append(winner)
    return parents


# Classic crossover, and fixing part that comes after it.
def crossover(parent1, parent2):
    n = len(parent1)

    # Performs crossover between two parent solutions to produce two offspring
    crossover_point = random.randint(1, n - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]

    # "Fixing" part, to ensure that the children are valid.
    set1 = set(child1)
    set2 = set(child2)
    # Find the difference
    diff1 = list(set2 - set1)
    diff2 = list(set1 - set2)
    for i in range(n - crossover_point):
        if child1[crossover_point + i] in parent1[:crossover_point]:
            child1[crossover_point + i] = diff1[0]
            diff1 = diff1[1:]
        if child2[crossover_point + i] in parent2[:crossover_point]:
            child2[crossover_point + i] = diff2[0]
            diff2 = diff2[1:]

    return child1, child2


def mutate(matching, mutation_rate):
    n = len(matching)
    if random.random() < mutation_rate:
        i, j = random.sample(range(n), 2)
        matching[i], matching[j] = matching[j], matching[i]
    return matching


# Calculates the average diversity among individuals in the population.
def measure_diversity(population):
    n = len(population)
    total_distance = 0
    for i in range(n):
        for j in range(i + 1, n):
            total_distance += sum(1 for x, y in zip(population[i], population[j]) if x != y)
    average_distance = total_distance / (n * (n - 1) / 2)
    return average_distance


# NOT USED!
def local_search(matching, preferences_males, preferences_females):
    best_matching = matching[:]
    best_fitness = fitness(best_matching, preferences_males, preferences_females)
    improved = True
    while improved:
        improved = False
        for i in range(len(matching)):
            for j in range(i + 1, len(matching)):
                new_matching = matching[:]
                new_matching[i], new_matching[j] = new_matching[j], new_matching[i]
                new_fitness = fitness(new_matching, preferences_males, preferences_females)
                if new_fitness > best_fitness:
                    best_matching = new_matching
                    best_fitness = new_fitness
                    improved = True
    return best_matching


# Performs a genetic algorithm in order to return the best matching,
# represented as a list where the index represents the male and the value represents the female matched.
def genetic_algorithm(preferences_males, preferences_females, population_size=180, generations=100, elitism_rate=0.04,
                      initial_mutation_rate=0.05, diversity_threshold=0.2, stagnation_limit=25, sharing_threshold=2.0):
    n = len(preferences_males)
    # population = create_initial_population(n, population_size)
    population = create_initial_population(n, population_size)
    mutation_rate = initial_mutation_rate
    max_fitnesses = []
    min_fitnesses = []
    avg_fitnesses = []
    best_fitness = -float('inf')
    stagnation_counter = 0
    use_fitness_sharing = False

    for generation in range(generations):
        fitnesses = [fitness(matching, preferences_males, preferences_females) for matching in population]
        new_population = []

        # Elitism
        num_elites = int(elitism_rate * population_size)
        elites = [x for _, x in sorted(zip(fitnesses, population), reverse=True)][:num_elites]
        new_population.extend(elites)

        # Select parents based on fitness sharing if early convergence is detected
        if use_fitness_sharing:
            # Calculate distances using hamming distance metric
            distances = calculate_distances(population, hamming_distance)
            adjusted_fitness = fitness_sharing(fitnesses, distances, sharing_threshold)

            # Note that it just a way to set variable for choosing what select parents I would use.
            # I do choose the same one in both cases, but it was very useful for trying all functions...
            parents_selection_func = select_parents_tournament
        else:
            adjusted_fitness = fitnesses
            parents_selection_func = select_parents_tournament

        # Tournament Selection for Parents
        while len(new_population) < population_size:
            # The parameter num_parents states the amount of parents that would return from the function.
            parents = parents_selection_func(population, adjusted_fitness, num_parents=2)

            child1, child2 = crossover(parents[0], parents[1])
            new_population.append(mutate(child1, mutation_rate))
            if len(new_population) < population_size:
                new_population.append(mutate(child2, mutation_rate))

        population = new_population

        # Measure diversity and adjust mutation rate
        diversity = measure_diversity(population)
        if diversity < diversity_threshold * n:
            mutation_rate *= 1.5  # Increase mutation rate if diversity is low
        else:
            mutation_rate = initial_mutation_rate  # Reset to initial mutation rate

        max_fitness = max(fitnesses)
        min_fitness = min(fitnesses)
        avg_fitness = sum(fitnesses) / len(fitnesses)
        max_fitnesses.append(max_fitness)
        min_fitnesses.append(min_fitness)
        avg_fitnesses.append(avg_fitness)

        if max_fitness > best_fitness:
            best_fitness = max_fitness
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        # if stagnation_counter > stagnation_limit and max_fitness > 0.72:
        if stagnation_counter > stagnation_limit:
            # Detect early convergence and switch to fitness sharing
            use_fitness_sharing = True
            stagnation_counter = 0

    best_matching = max(population, key=lambda x: fitness(x, preferences_males, preferences_females))

    # Plot fitness statistics
    plt.figure(figsize=(10, 6))
    plt.plot(max_fitnesses, label='Max Fitness')
    plt.plot(min_fitnesses, label='Min Fitness')
    plt.plot(avg_fitnesses, label='Avg Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Fitness over Generations')
    plt.legend()

    final_best_fitness = max_fitness
    final_best_matching = best_matching
    annotation_text = f'Best Matching: {final_best_matching}\nFitness: {final_best_fitness:.2f}'

    # Calculate the annotation position in the center of the plot
    plt.annotate(annotation_text,
                 xy=(generations - 100, final_best_fitness),
                 xytext=(generations / 2, final_best_fitness - (max_fitnesses[0] - min_fitnesses[0]) * 2.1),
                 fontsize=10,
                 bbox=dict(facecolor='white', alpha=0.8),
                 horizontalalignment='center',
                 verticalalignment='center')

    plt.show()

    # Return the best matching
    return best_matching


if __name__ == "__main__":
    filename = 'GA_input.txt'
    n, preferences_males, preferences_females = read_preferences(filename)

    best_matching = genetic_algorithm(preferences_males, preferences_females, population_size=180, generations=100,
                                      elitism_rate=0.04, initial_mutation_rate=0.05)
    # Note that the "flush=True" is in order that the exe file would print those prints into the terminal.
    print(f'Best Matching: {best_matching}', flush=True)
    print("Fitness rank:", fitness(best_matching, preferences_males, preferences_females), flush=True)
