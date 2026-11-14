from genetics import create_human


def create_population(size, allele_probs=None):
  if not isinstance(size,int) or size < 1:
    raise ValueError("Population size must be a positive integer")
  
  population = []
  for _ in range(size):
    population.append(create_human(allele_probs=allele_probs))

  return population