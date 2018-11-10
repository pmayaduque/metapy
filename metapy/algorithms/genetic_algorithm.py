import numpy as np


class BaseGeneticAlgorithm(object):
    """Base class for the genetic algorithm. The user inherits from this class and specifies the evolutionary methods."""

    def __init__(self, population_size, generations, mutation_rate, selection_rate, survival_mode='normal', optimization_mode='minimize'):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.selection_rate = selection_rate
        self.survival_mode = survival_mode
        self.optimization_mode = optimization_mode

        self.population = None

    def init_population(self):
        """Initializes Population
        """

    def fitness(self, candidate):
        """Returns fitness for a given candidate. A 

        Args:
            candidate (Vector): candidate from population

        Returns:
            float: scalar fitness value
        """
        pass

    def crossover(self, candidates):
        """Performs Crossover of given candidates (usually two)

        Args:
            candidates (List[Vector]): two or more candidates from population
        Returns:
            Vector: new candidate
        """
        pass

    def mutation(self, candidate):
        """Performs mutation of candidate

        Args:
            candidate (Vector): candidate from population
        """
        pass

    def health(self, candidate):
        """Checks health of candidate and returns a healthy candidate

        Args:
            candidate (Vector): candidate from population
        """
        pass
