import numpy as np
import metapy as mp
from metapy.algorithms import GeneticAlgorithm


class RastriginAlgorithm(GeneticAlgorithm):
    def __init__(self, generations=20, mutation_rate=0.2, population_size=100, elitism=0, minimize=True):
        super().__init__(generations, mutation_rate, population_size, elitism=elitism, minimize=minimize)
        self.low = -5.12 * np.ones(10)
        self.high = 5.12 * np.ones(10)
        self.population = [np.random.uniform(-5.12, 5.12, 10) for i in range(self.population_size)]
        self.survival_rate = 0.2

    def fitness(self, candidate):
        A = 10
        f = A * len(candidate) + np.sum(candidate**2 - A * np.cos(2*np.pi*candidate))
        return f
    
    def crossover(self, candidates):
        return mp.crossover.arithmetic_crossover(candidates)
    
    def selection(self, fitness):
        return mp.selection.sample_from_fittest(self.population, fitness, self.selection_size, self.survival_rate)

    def mutation(self, candidate):
        if np.random.uniform() > self.mutation_rate:
            return mp.mutation.gauss_mutation(candidate)
        else:
            return candidate

def main():
    ga = RastriginAlgorithm()
    res = ga.optimize()
    print(res)

if __name__ == '__main__':
    main()
