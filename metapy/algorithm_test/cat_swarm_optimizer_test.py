from math import pow
from random import random
from metapy.algorithms.cat_swarm_optimization import Cat, CatSwarmOptimization


def rosenbrock_fitness(cat):
    try:
        return pow((1 - cat.position[0]), 2) + 100 * pow(cat.position[1] - pow(cat.position[0], 2), 2)
    except(OverflowError):
        return float('inf')

def build_population():
    catlist = []
    for i in range(20):
        catlist.append(
            Cat([random() * 3, random() * 3], [random()*0.2, random()*0.2],
                0.2, rosenbrock_fitness)
        )
    return catlist



cso = CatSwarmOptimization(0.2, 20, [1, 1], 2, True,
                           2,
                           rosenbrock_fitness, 0, 2500, True)
cso.init_population(build_population())

for i in range(20):
    currentcats = cso.optimize(50)
    print("Best cat evaluates to: " + str(min(currentcats)))
