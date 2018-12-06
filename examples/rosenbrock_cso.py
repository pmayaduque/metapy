from math import pow
from random import random
from metapy.algorithms.cat_swarm_optimization import Cat, CatSwarmOptimization

# ToDo: Rosenbrock function does not look a single bit like it should be ...
#       so this one needs some work
def rosenbrock_fitness(cat):
    return pow((1 - cat.position[0]), 2) + 100 * pow(cat.position[1] - pow(cat.position[0], 2), 2)


class RosenbrockCSO(CatSwarmOptimization):
    def fitness(self, cat):
        return rosenbrock_fitness(cat)


cso = RosenbrockCSO(0.2, 20, [1, 1], 2, True,
                    2, [0.2, 0.2],
                    0, 2500, True)
cso.init_population(5)


currentcats = cso.optimize(50)
print("Best cat evaluates to: " + str(min(currentcats)))
