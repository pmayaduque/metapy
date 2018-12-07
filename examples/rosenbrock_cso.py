from math import pow
from random import random
from metapy.algorithms.cat_swarm_optimization import Cat, CatSwarmOptimization

def rosenbrock_fitness(cat):
    return pow((1 - cat.position[0]), 2) + 100 * pow(cat.position[1] - pow(cat.position[0], 2), 2)


class RosenbrockCSO(CatSwarmOptimization):
    def fitness(self, cat):
        return rosenbrock_fitness(cat)


cso = RosenbrockCSO(0.2, 10, [0.5, 0.5], 2, True,
                    2, [0.2, 0.2],
                    2500, 0, True)
cso.init_population(5)


currentcats = cso.optimize(500)
print("Best cat evaluates to: " + str(min(currentcats)))
