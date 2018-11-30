import metapy
import numpy as np
import math
from metapy.algorithms import GeneticAlgorithm, SimulatedAnnealing, AntColonyAlgorithm
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
                distance += self.distance_matrix[now][next_stop]
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


class AnnealingTSP(SimulatedAnnealing):
    def __init__(self, distance_matrix, state=None, init_temperature=None, minimize=True):
        self.distance_matrix = distance_matrix
        return super().__init__(state=state, init_temperature=init_temperature, minimize=minimize)
    
    def energy(self, state):
        now = state[0]
        distance = 0.0
        route = (point for point in state[1:])
        while True:
            try:
                next_stop = next(route)
                distance += self.distance_matrix[now][next_stop]
                now = next_stop
            except StopIteration:
                break
        return distance
    
    def alter(self, state):
        return metapy.mutation.swap_mutation(state)


class AntColonyTSP(AntColonyAlgorithm):
    def __init__(self, distance_matrix, colony_size, evaporation=0.01, pheromone_factor=1.0, elitism=None):
        return super().__init__(distance_matrix, colony_size, evaporation=evaporation, pheromone_factor=pheromone_factor, elitism=elitism)


def distance(a, b):
    """Calculates distance between two latitude-longitude coordinates."""
    R = 3963  # radius of Earth (miles)
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    return math.acos(math.sin(lat1) * math.sin(lat2) +
                     math.cos(lat1) * math.cos(lat2) * math.cos(lon1 - lon2)) * R


if __name__ == "__main__":
    # latitude and longitude for the twenty largest U.S. cities
    cities = {
        'New York City': (40.72, 74.00),
        'Los Angeles': (34.05, 118.25),
        'Chicago': (41.88, 87.63),
        'Houston': (29.77, 95.38),
        'Phoenix': (33.45, 112.07),
        'Philadelphia': (39.95, 75.17),
        'San Antonio': (29.53, 98.47),
        'Dallas': (32.78, 96.80),
        'San Diego': (32.78, 117.15),
        'San Jose': (37.30, 121.87),
        'Detroit': (42.33, 83.05),
        'San Francisco': (37.78, 122.42),
        'Jacksonville': (30.32, 81.70),
        'Indianapolis': (39.78, 86.15),
        'Austin': (30.27, 97.77),
        'Columbus': (39.98, 82.98),
        'Fort Worth': (32.75, 97.33),
        'Charlotte': (35.23, 80.85),
        'Memphis': (35.12, 89.97),
        'Baltimore': (39.28, 76.62)
    }
    # initial state, a randomly-ordered itinerary
    init_state = list(cities.keys())
    shuffle(init_state)

    # create a distance matrix
    distance_matrix = {}
    for ka, va in cities.items():
        distance_matrix[ka] = {}
        for kb, vb in cities.items():
            if kb == ka:
                distance_matrix[ka][kb] = 0.0
            else:
                distance_matrix[ka][kb] = distance(va, vb)
    population = []
    for i in range(200):
        l = [i for i in list(cities.keys())]
        shuffle(l)
        population.append(l)
    tsp = GeneticTSP(50, 0.2, 200, distance_matrix, elitism=10)
    tsp.population = population
    res1 = tsp.optimize(6)  
    print(res1.best_fitness[-1])
    
    state = [i for i in list(cities.keys())]
    shuffle(state)
    tsp = AnnealingTSP(distance_matrix, state, init_temperature=100)
    res2 = tsp.optimize(2000)
    print(res2.energies[-1])

    city_keys = list(cities.keys())
    np_distance_matrix = np.zeros((len(city_keys), len(city_keys)))
    for a, name_a in enumerate(city_keys):
        for b, name_b in enumerate(city_keys):
            if name_a == name_b:
                continue
            else:
                np_distance_matrix[a, b] = distance_matrix[name_a][name_b]

    tsp = AntColonyTSP(np_distance_matrix, 100, evaporation=0.2, pheromone_factor=10)
    res3 = tsp.optimize(100, parallel=True)
    print(res3.solution)
