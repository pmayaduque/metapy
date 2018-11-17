import metapy
import numpy as np
from metapy.algorithms import GeneticAlgorithm
from random import shuffle


class GeneticTSP(GeneticAlgorithm):
    """Genetic Algorithm example for the Traveling Salesman Problem. Chromosomes represent the order of cities. Distances are provided by a distance matrix.
    """

    def __init__(self, generations, mutation_rate, population_size, distance_matrix, elitism=0, minimize=True):
        self.distance_matrix = distance_matrix
        return super().__init__(generations, mutation_rate, population_size, elitism=elitism, minimize=minimize)

    def fitness(self, candidate):
        now = candidate[0]
        distance = 0.0
        route = (point for point in candidate[1:])
        while True:
            try:
                next_stop = next(route)
                distance += self.distance_matrix[now, next_stop]
                now = next_stop
            except StopIteration:
                break
        return distance

    def crossover(self, candidates):
        return metapy.crossover.order_based_crossover(candidates)

    def mutation(self, candidate):
        return metapy.mutation.swap_mutation(candidate)

    def selection(self, fitness):
        return metapy.selection.rank_based_selection(self.population, fitness, self.selection_size)


def make_distance_matrix(points):
    distance_matrix = np.zeros((len(points), len(points)))
    for i, a in enumerate(points):
        for j, b in enumerate(points):
            distance_matrix[i, j] = np.linalg.norm(a-b)
    return distance_matrix

if __name__ == "__main__":
    points = [np.array((np.random.uniform(0, 10), np.random.uniform(0, 10))) for i in range(20)]
    distance_matrix = make_distance_matrix(points)
    population = []
    for i in range(200):
        l = [i for i in range(20)]
        shuffle(l)
        population.append(l)
    tsp = GeneticTSP(50, 0.2, 200, distance_matrix, elitism=10)
    tsp.population = population

    print(tsp.optimize(6))
