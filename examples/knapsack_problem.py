import numpy as np
import metapy as mp
from metapy.algorithms import GeneticAlgorithm, SimulatedAnnealing


class GeneticKnapsackAlgorithm(GeneticAlgorithm):
    def __init__(self, values, weights):
        super().__init__(generations=20, mutation_rate=0.1,
                         population_size=10, elitism=0, minimize=False)
        self.population = [np.random.randint(
            0, 2, 10) for i in range(self.population_size)]
        self.values = values
        self.weights = weights

    def fitness(self, candidate):
        W = 10
        total_weight = 0
        fitness = 0
        for i, bit in enumerate(candidate):
            if bit == 1:
                total_weight += self.weights[i]
                fitness += self.values[i]

        if total_weight > W:
            fitness -= 3 * (total_weight - W)

        return fitness

    def crossover(self, candidates):
        return mp.crossover.uniform_crossover(candidates)

    def mutation(self, candidate):
        if np.random.uniform() > self.mutation_rate:
            return mp.mutation.bit_flip_mutation(candidate)
        else:
            return candidate

    def selection(self, fitness):
        return mp.selection.rank_based_selection(self.population, fitness, self.selection_size, number_of_parents=2, minimize=False)


class SimulatedAnnealingKnapsack(SimulatedAnnealing):
    def __init__(self, values=None, weights=None, state=None, init_temperature=None, minimize=True):
        self.values = values
        self.weights = weights
        return super().__init__(state=state, init_temperature=init_temperature, minimize=minimize)

    def energy(self, state):
        W = 10
        total_weight = 0
        fitness = 0
        for i, bit in enumerate(state):
            if bit == 1:
                total_weight += self.weights[i]
                fitness += self.values[i]

        if total_weight > W:
            fitness -= 3 * (total_weight - W)

        return fitness

    def alter(self, state):
        return mp.mutation.bit_flip_mutation(state)


def main():
    values = [np.random.randint(0, 10) * 10 for i in range(10)]
    weights = [np.random.randint(0, 10) for i in range(10)]

    ga = GeneticKnapsackAlgorithm(values, weights)
    res = ga.optimize()
    print(res)

    sa = SimulatedAnnealingKnapsack(values, weights, state=np.random.randint(
        0, 2, 10), init_temperature=100, minimize=False)
    res = sa.optimize()
    print(res)


if __name__ == '__main__':
    main()
