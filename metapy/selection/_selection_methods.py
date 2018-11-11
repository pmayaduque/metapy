import random
import numpy as np


def sample_from_fittest(population, fitness, size, survival_rate, reverse=False, number_of_parents=2):
    """Sorts population by the fitness value and samples parents randomly from the fittest chromosomes.
    
    Args:
        population (List(Vector)): list of candidates
        fitness (List(float)): list of fitness values corresponding to the candidates
        size (int): how many parents to sample
        survival_rate (float): between 0 and 1, percentage of surviving 
        reverse (bool, optional): Defaults to False. Determines if lower or higher is better
        number_of_parents (int, optional): Defaults to 2. Size of couples for crossover
    
    Returns:
        List(List(Vector)): list of length size containing parent chromosomes
    """
    fitness_and_pop = list(zip(fitness, population))
    fitness_and_pop = sorted(fitness_and_pop, reverse=reverse, key=lambda x: x[0])
    _, pop = zip(*fitness_and_pop)
    survivers_idx = int(len(pop) * survival_rate)
    survivers = pop[:survivers_idx]
    parents = [[random.choice(survivers) for _ in range(number_of_parents)] for i in range(size)]
    return parents


def rank_based_selection(population, fitness, size, reverse=False, number_of_parents=2):
    """Sorts population by fitness and then selects parents with a probability according to their rank. Better parents have higher probability
    
    Args:
        population (List(Vector)): list of candidates
        fitness (List(float)): list of fitness values corresponding to the candidates
        size (int): how many parents to sample
        reverse (bool, optional): Defaults to False. Determines if lower or higher is better
        number_of_parents (int, optional): Defaults to 2. Size of couples for crossover
    
    Returns:
        List(List(Vector)): list of length size containing parent chromosomes
    """
    fitness_and_pop = list(zip(fitness, population))
    fitness_and_pop = sorted(fitness_and_pop, reverse=reverse, key=lambda x: x[0])
    _, pop = zip(*fitness_and_pop)
    weights = np.arange(len(pop), 0, -1)
    weights = weights / np.sum(weights)
    parents = [random.choices(pop, k=number_of_parents, weights=weights) for i in range(size)]
    return parents
