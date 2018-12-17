import numpy as np
from multiprocessing import Pool
from heapq import nlargest, nsmallest
from metapy.algorithms._base import Algorithm, Result


class GeneticAlgorithm(Algorithm):
    """Class for the Genetic Algorithm

    Args:
        generations (int): maximum number of generations
        mutation_rate (float): value between 0 and 1 for the probability of mutation
        population_size (int): size of each population
        elitism (int): Default is 0, number of best chromosomes which are allowed to survive each generation
        minimize (bool): if True, lower function value is better
    """

    def __init__(self, mutation_rate=0.2, population_size=100, elitism=0, minimize=True):
        Algorithm.__init__(self)
        self.mutation_rate = mutation_rate
        self.population_size = population_size
        self.elitism = elitism
        self.minimize = minimize

        self.selection_size = self.population_size - self.elitism
        self.population = None
        self.function_calls = {'fitness': 0, 'crossover': 0, 'mutation': 0, 'selection': 0}

    def optimize(self, generations=np.inf, max_time=np.inf):
        """Performs optimization. The optimization follows three steps:
        - for current population calculate fitness
        - select chromosomes with best fitness values with higher propability as parents
        - use parents for reproduction (crossover and mutation)
        - repeat until max number of generation is reached
        Args:
            generations (int): Max Iterations
            max_time (float): maximum time
        Returns:
            Result
        """
        self.max_iterations = generations
        self.max_time = max_time

        if np.isinf(self.max_iterations) and np.isinf(self.max_time):
            raise TypeError('Expect either max_iterations or max_time to be not inf.')

        if self.population is None:
            self.init_population()

        res = Result()
        res.optimizer_settings = {'mutation_rate': self.mutation_rate, 'population_size': self.population_size,
                                  'elitism': self.elitism, 'generations': self.max_iterations, 'time limit': self.max_time}
        self.start()
        while not self.has_finished():
            # calculate fitness for each candidate in the population
            fitness = [self._fitness(candidate) for candidate in self.population]

            res.averaged_progress.append(sum(fitness) / len(fitness))

            if self.minimize:
                res.best_progress.append(min(fitness))
            else:
                res.best_progress.append(max(fitness))

            # get parents
            parents = self._selection(fitness)

            # perform crossover
            children = [self._crossover(p) for p in parents]

            # perform mutation
            children = [self._mutate(child) for child in children]

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

            self.iterations += 1

        res.solution = min(self.population, key=self.fitness)
        res.function_calls = self.function_calls
        return res

    def init_population(self):
        """Initializes Population
        """
        raise NotImplementedError("Either provide an initial population or an init_population method.")
    
    def _fitness(self, candidate):
        """internal method for calling the fitness method"""
        self.function_calls['fitness'] += 1
        return self.fitness(candidate)

    def fitness(self, candidate):
        """Returns fitness for a given candidate. A 

        Args:
            candidate (Vector): candidate from population

        Returns:
            float: scalar fitness value
        """
        raise NotImplementedError("No fitness method found.")
    
    def _crossover(self, candidates):
        """Internal method for calling the crossover method"""
        self.function_calls['crossover'] += 1
        return self.crossover(candidates)

    def crossover(self, candidates):
        """Performs Crossover of given candidates (usually two)

        Args:
            candidates (List[Vector]): two or more candidates from population
        Returns:
            Vector: new candidate
        """
        raise NotImplementedError("No crossover method found.")
    
    def _selection(self, fitness):
        """Internal method for calling the selection method"""
        self.function_calls['selection'] += 1
        return self.selection(fitness)
        
    def selection(self, fitness):
        """Performs selection. 

        Args:
            fitness (List(float))

        Raises:
            List(List(candidate)) - selected as parents
        """
        raise NotImplementedError("No selection method found.")
        
    def _mutate(self, candidate):
        """Internal method for calling the mutation method"""
        if np.random.uniform() > self.mutation_rate:
            self.function_calls['mutation'] += 1
            return self.mutation(candidate)
        else:
            return candidate

    def mutation(self, candidate):
        """Performs mutation of candidate

        Args:
            candidate (Vector): candidate from population
        """
        raise NotImplementedError("No mutation method found.")

    def health(self, candidate):
        """Checks health of candidate and returns a healthy candidate

        Args:
            candidate (Vector): candidate from population
        """
        # TODO: Integrate health into algorithm
        raise NotImplementedError

# TODO: Implement an easy method for setting up a genetic algorithm
