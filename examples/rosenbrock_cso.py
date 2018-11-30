from math import pow
from random import random
from metapy.algorithms.cat_swarm_optimization import Cat, CatSwarmOptimization


def rosenbrock_fitness(cat):
    try:
        fit = pow((1 - cat.position[0]), 2) + 100 * pow(cat.position[1] - pow(cat.position[0], 2), 2)
        return fit
    except(OverflowError):
        return float('inf')


class RosenbrockCSO(CatSwarmOptimization):
    def fitness(self, cat):
        return rosenbrock_fitness(cat)


cso = RosenbrockCSO(0.2, 20, [1, 1], 2, True,
                    2, 0, 2500, True)
cso.init_population(10)

for i in range(20):
    currentcats = cso.optimize(50)
    print("Best cat evaluates to: " + str(min(currentcats)))
