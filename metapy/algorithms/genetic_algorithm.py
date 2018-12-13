import numpy as np
from multiprocessing import Pool
from heapq import nlargest, nsmallest


class GeneticOptimizationResult(object):
    def __init__(self, mutation_rate, population_size):
        self.mutation_rate = mutation_rate
        self.population_size = population_size
        self.solution = None
        self.best_fitness = []
        self.average_fitness = []

    def __str__(self):
        d = {'solution': self.solution,
             'best_fitness': self.best_fitness,
             'average_fitness': self.average_fitness,
             'population_size': self.population_size,
             'mutation_rate': self.mutation_rate}
        return str(d)


class GeneticAlgorithm(object):
    """Class for the Genetic Algorithm

    Args:
        generations (int): maximum number of generations
        mutation_rate (float): value between 0 and 1 for the probability of mutation
        population_size (int): size of each population
        elitism (int): Default is 0, number of best chromosomes which are allowed to survive each generation
        minimize (bool): if True, lower function value is better
    """

    def __init__(self, generations, mutation_rate, population_size, elitism=0, minimize=True):
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.population_size = population_size
        self.elitism = elitism
        self.minimize = minimize

        self.selection_size = self.population_size - self.elitism
        self.population = None

    def optimize(self, number_of_processes=1):
        """Performs optimization. The optimization follows three steps:
        - for current population calculate fitness
        - select chromosomes with best fitness values with higher propability as parents
        - use parents for reproduction (crossover and mutation)
        - repeat until max number of generation is reached

            number_of_processes (int, optional): Defaults to 1. Parallel computation of fitness values and reproduction is allowed

        Returns:
            GeneticOptimizationResult
        """

        if self.population is None:
            self.init_population()

        res = GeneticOptimizationResult(
            self.mutation_rate, self.population_size)
        generation = 0

        # TODO let the user choose between different stop criteria
        while generation < self.generations:
            # calculate fitness for each candidate in the population
            with Pool(number_of_processes) as p:
                fitness = p.map(self.fitness, self.population)

            res.average_fitness.append(sum(fitness) / len(fitness))

            if self.minimize:
                res.best_fitness.append(min(fitness))
            else:
                res.best_fitness.append(max(fitness))

            # get parents
            parents = self.selection(fitness)

            # perform crossover
            with Pool(number_of_processes) as p:
                children = p.map(self.crossover, parents)

            # perform mutation
            with Pool(number_of_processes) as p:
                children = p.map(self._mutate, children)

            if self.elitism > 0:
                if self.minimize:
                    surviving_elite = nsmallest(self.elitism, list(
                        range(len(self.population))), key=lambda x: fitness[x])
                else:
                    surviving_elite = nlargest(self.elitism, list(
                        range(len(self.population))), key=lambda x: fitness[x])
                surviving_elite = [self.population[i] for i in surviving_elite]
                self.population = surviving_elite + children
            else:
                self.population = children

            generation += 1

        res.solution = min(self.population, key=self.fitness)
        return res

    def init_population(self):
        """Initializes Population
        """
        raise NotImplementedError

    def fitness(self, candidate):
        """Returns fitness for a given candidate. A 

        Args:
            candidate (Vector): candidate from population

        Returns:
            float: scalar fitness value
        """
        raise NotImplementedError

    def crossover(self, candidates):
        """Performs Crossover of given candidates (usually two)

        Args:
            candidates (List[Vector]): two or more candidates from population
        Returns:
            Vector: new candidate
        """
        raise NotImplementedError

    def selection(self, fitness):
        """Performs selection. 

        Args:
            fitness (List(float))

        Raises:
            List(List(candidate)) - selected as parents
        """
        raise NotImplementedError
        
    def _mutate(self, candidate):
        """Internal method for calling the mutation method"""
        if np.random.uniform() > self.mutation_rate:
            return self.mutation(candidate)
        else:
            return candidate

    def mutation(self, candidate):
        """Performs mutation of candidate

        Args:
            candidate (Vector): candidate from population
        """
        raise NotImplementedError

    def health(self, candidate):
        """Checks health of candidate and returns a healthy candidate

        Args:
            candidate (Vector): candidate from population
        """
        raise NotImplementedError

# TODO: Implement an easy method for setting up a genetic algorithm
