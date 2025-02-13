# Genetic Algorithm for Stable Marriage Problem

## Overview
This implementation uses a genetic algorithm to solve the Stable Marriage Problem, where n males and n females need to be matched based on their preference lists. Each person ranks all members of the opposite gender from most preferred to least preferred.

## Approach
The solution employs evolutionary computation principles through a genetic algorithm:

### Representation
- Each solution (chromosome) is represented as a list where indices represent males and values represent females.
- The population consists of multiple valid matchings.

### Genetic Operations
- **Crossover**: Combines parts of two parent matchings to create offspring.
- **Mutation**: Randomly swaps pairs in a matching to introduce variation.
- **Selection**: Tournament selection to choose parents for next generation.
- **Elitism**: Preserves best solutions across generations.

### Fitness Function
Evaluates matching quality based on preference satisfaction:
- Higher scores for matches closer to top preferences.
- Normalized to range [0,1].
- Considers preferences of both males and females.

### Features
- Dynamic mutation rate adjustment.
- Niching strategy to maintain population diversity.
- Fitness sharing to prevent premature convergence.
- Population statistics visualization.

 ## Algorithm Details

The genetic algorithm implementation includes several key features to ensure effective exploration of the solution space:

### Parameters
- Population size (remains static throughout execution).
- Number of generations.
- Elitism rate (percentage of best solutions preserved between generations).
- Initial mutation rate (dynamically adjustable).
- Diversity threshold (for population diversity maintenance).
- Stagnation limit (for early convergence detection).
- Sharing threshold (for niching implementation).

### Process Flow
1. **Initial Population**: Creates a random initial population of valid matchings.

2. **Evolution Loop**:
   - Calculate fitness scores for all matchings.
   - Preserve elite solutions.
   - Apply niching if early convergence detected:
     - Uses Hamming distance to measure solution similarity.
     - Adjusts fitness scores to promote diversity.
   - Create new generation:
     - Select parents through tournament selection.
     - Perform crossover at random points.
     - Apply mutation with dynamic rate.
   - Measure population diversity:
     - Increase mutation rate if diversity falls below threshold.
   - Track convergence metrics.

3. **Dynamic Adaptation**:
   - Monitors solution quality and population diversity.
   - Adjusts mutation rates when needed.
   - Implements niching strategy for local optima escape.
   - Maintains elitism for consistent improvement.

This implementation balances exploration and exploitation through dynamic parameter adjustment and diversity preservation techniques. 

## Results
Various combinations of population size and generation count were tested:
- **Population: 180, Generations: 100 → Fitness: 0.927**
  
  ![POP=180_GEN=100](https://github.com/user-attachments/assets/59bb685e-418c-4f5a-83b4-266ee43a77cf)
- **Population: 150, Generations: 120 → Fitness: 0.912**
  
  ![POP=150_GEN=120](https://github.com/user-attachments/assets/38fd4990-66d9-403a-b524-8772678272ae)
- **Population: 100, Generations: 180 → Fitness: 0.919**
  
  ![POP=100_GEN=180](https://github.com/user-attachments/assets/54a393e7-1f07-4c2c-9f2f-483e5fecb073)

**The algorithm consistently achieved high-quality matchings (fitness > 0.90) regardless of the specific parameter configuration. Even with fewer generations (40-50), the algorithm maintained good performance with fitness scores around 0.88.**
![POP=180_GEN=40](https://github.com/user-attachments/assets/7e6920fa-d3af-4ebc-9c58-dc9573ee1383)
![POP=100_GEN=50](https://github.com/user-attachments/assets/248f9e39-2c91-41c7-983d-1cf04a37c52d)

