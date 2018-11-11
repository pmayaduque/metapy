import numpy as np
from multiprocessing import Pool
from heapq import nlargest, nsmallest
from metapy.crossover import uniform_crossover, one_point_crossover
from metapy.selection import rank_based_selection, sample_from_fittest
from metapy.mutation import bit_flip_mutation, rand_int_mutation



class GeneticAlgorithm(object):
    """Base class for the genetic algorithm. The user inherits from this class and specifies the evolutionary methods."""

    def __init__(self, generations, mutation_rate, selection_size, elitism=0, minimize=True):
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.selection_size = selection_size
        self.elitism = elitism
        self.minimize = minimize

        self.survival_rate = None
        self.population = None

    def optimize(self, number_of_processes=1):
        if self.population is None:
            self.init_population()

        res = {
            'avg fitness': [],
            'best fitness': [],
            'x': None
        }
        generation = 0
        while generation < self.generations:
            # calculate fitness for each candidate in the population
            with Pool(number_of_processes) as p:
                fitness = p.map(self.fitness, self.population)

            res['avg fitness'].append(sum(fitness) / len(fitness))

            if self.minimize:
                res['best fitness'].append(min(fitness))
            else:
                res['best fitness'].append(max(fitness))

            # get parents
            parents = self.selection(fitness)

            # perform crossover
            with Pool(number_of_processes) as p:
                children = p.map(self.crossover, parents)

            # perform mutation
            with Pool(number_of_processes) as p:
                children = p.map(self.mutation, children)

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

        res['x'] = min(self.population, key=self.fitness)
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

    # crossover methods
    def uniform_crossover(self, candidates):
        return uniform_crossover(candidates)
    
    def one_point_crossover(self, candidates):
        return one_point_crossover(candidates)

    # selection methods
    def rank_based_selection(self, fitness, number_of_parents=2):
        return rank_based_selection(self.population, fitness, self.selection_size, reverse=not self.minimize, number_of_parents=number_of_parents)
    
    def sample_from_fittest(self, fitness, number_of_parents=2):
        return sample_from_fittest(self.population, fitness, self.selection_size, survival_rate=self.survival_rate, reverse=not self.minimize, number_of_parents=number_of_parents)

    # mutation methods
    def bit_flip_mutation(self, candidate):
        return bit_flip_mutation(candidate)

    def rand_int_mutation(self, candidate, low=None, high=None):
        if low is None:
            low = min(candidate)
        if high is None:
            high = max(candidate)
        
        return rand_int_mutation(candidate, low, high)


# TODO: Add Genetic Algorithm result class
# TODO: Add Easy Genetic Algorithm setup