import numpy as np

from genetics import reproduce
from analytics import trait_frequency


def calculate_fitness(human, scenario, positive_fitness=1.5, negative_fitness=0.5):
  if scenario == "neutral":
    fitness = 1.0

  elif scenario == "genetic_drift":
    fitness = 1.0

  elif scenario == "positive":
    if human['has_large_hand']:
      fitness = positive_fitness
    else:
      fitness = 1.0

  elif scenario == "negative":
    if human['has_large_hand']:
      fitness = negative_fitness
    else:
      fitness = 1.0
      
  else:
        raise ValueError("Unknown scenario")

  return fitness


def create_next_generation(population, scenario, positive_fitness=1.5, negative_fitness=0.5):
  new_gen = []

  fitness_values = []

  for human in population:
    fitness_values.append(calculate_fitness(human, scenario, positive_fitness, negative_fitness))

  human_prob = np.array(fitness_values, dtype=float)
  human_prob = human_prob / human_prob.sum()

  for _ in range(len(population)):
    parents = np.random.choice(
        len(population),
        size=2,
        replace=True,
        p=human_prob
    )

    parent1 = population[parents[0]]
    parent2 = population[parents[1]]

    new_gen.append(reproduce(parent1, parent2))

  return new_gen


def simulate_evolution(population, scenario, generations, positive_fitness=1.5, negative_fitness=0.5):
  gen_freq = []

  for _ in range(generations):
    new_population = create_next_generation(population, scenario, positive_fitness, negative_fitness)
    gen_freq.append(trait_frequency(new_population))
    population = new_population

  return gen_freq