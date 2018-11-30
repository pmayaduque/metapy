import numpy as np
import random
from os import cpu_count
from multiprocessing import Pool

class AntColonyOpimizaionResult(object):
    def __init__(self, colony_size, evaporation, pheromone_factor):
        self.colony_size = colony_size
        self.evaporation = evaporation
        self.pheromone_factor = pheromone_factor
        self.solution = None
        self.best_distances = []
        self.average_distances = []


class AntColonyAlgorithm(object):
    def __init__(self, distance_matrix, colony_size, evaporation=0.01, pheromone_factor=1.0, elitism=None):
        self.distance_matrix = distance_matrix
        self.colony_size = colony_size
        self.evaporation = evaporation
        self.pheromone_factor = pheromone_factor
        self.elitism = elitism

        self.epsilon = 1e-5
        self.pheromone_matrix = None
    
    def optimize(self, maxiter=1000, parallel=False):
        if parallel:
            number_of_processes = max(cpu_count()-1, 1)
        else:
            number_of_processes = 1

        self.pheromone_matrix = np.ones_like(self.distance_matrix) * self.epsilon
        best_tour = [i for i in range(self.distance_matrix.shape[0])]
        random.shuffle(best_tour)
        
        result = AntColonyOpimizaionResult(self.colony_size, self.evaporation, self.pheromone_factor)

        for i in range(maxiter):
            self.pheromone_matrix = (1 - self.evaporation) * self.pheromone_matrix
            with Pool(number_of_processes) as p:
                tours = p.map(self.make_tour, tuple([i for i in range(self.colony_size)]))
                distances = p.map(self.distance, tours)

            # tours = [self.make_tour() for i in range(self.colony_size)]
            # distances = [self.distance(tour) for tour in tours]
            self.place_pheromone(tours, distances)

            best_tour, best_distance = min(zip(tours, distances), key=lambda x: x[1])
            average_distance = np.sum(distances) / len(distances)
            result.best_distances.append(best_distance)
            result.average_distances.append(average_distance)

        result.solution = (best_tour, best_distance)
        return result

    def make_tour(self, *args):
        remaining_points = [i for i in range(self.distance_matrix.shape[0])]
        random.shuffle(remaining_points)
        remaining_points = set(remaining_points)

        tour = []
        while len(remaining_points) > 0:
            if len(tour) == 0:
                tour.append(remaining_points.pop())
                continue
            last_point = tour[-1]
            points_and_pheromone = [(point, self.pheromone_matrix[last_point, point]) for point in remaining_points]
            points, pheromone = zip(*points_and_pheromone)
            pheromone = pheromone / np.sum(pheromone)  #normalize pheromone to have sum=1

            next_point = int(np.random.choice(points, size=1, p=pheromone))
            tour.append(next_point)
            remaining_points.remove(next_point)

        return tour
    
    def distance(self, tour):
        distance = 0.0
        current_pos = tour[0]
        for i in range(1, len(tour)):
            next_pos = tour[i]
            distance += self.distance_matrix[current_pos, next_pos]
            current_pos = next_pos

        return distance
    
    def place_pheromone(self, tours, distances):
        for tour, distance in zip(tours, distances):
            additional_pheromone = np.zeros_like(self.pheromone_matrix)
            for a, b in zip(tour[1:], tour[:-1]):
                pheromone = self.pheromone_factor / distance
                additional_pheromone[a, b] = pheromone
            
            self.pheromone_matrix += additional_pheromone
