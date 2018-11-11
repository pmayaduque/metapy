# TODO Show usage of Genetic algorithm on the example of the Knapsack Problem
import numpy as np
from metapy.algorithms import BaseGeneticAlgorithm
from metapy.crossover import uniform_crossover
from metapy.mutation import bit_flip_mutation
from metapy.selection import rank_based_selection


class KnapsackAlgorithm(BaseGeneticAlgorithm):
    def __init__(self):
        super().__init__(generations=20, mutation_rate=0.1, selection_size=10, elitism=0, minimize=False)
        self.population = [np.random.randint(0, 2, 10) for i in range(10)]
    
    def fitness(self, candidate):
        values = [np.random.randint(0, 10) * 10 for i in range(10)]
        weights = [np.random.randint(0, 10) for i in range(10)]

        W = 10

        total_weight = 0
        fitness = 0
        for i, bit in enumerate(candidate):
            if bit == 1:
                total_weight += weights[i]
                fitness += values[i]
        
        if total_weight > W:
            fitness -= 10 * (total_weight - W)
        
        return fitness
    
    def crossover(self, candidates):
        return uniform_crossover(candidates)

    def mutation(self, candidate):
        return bit_flip_mutation(candidate)
    
    def selection(self, fitness):
        return rank_based_selection(self.population, fitness, self.selection_size, reverse=True, number_of_parents=2)


def main():
    ga = KnapsackAlgorithm()
    res = ga.optimize(3)
    print(res['x'])

if __name__ == '__main__':
    main()