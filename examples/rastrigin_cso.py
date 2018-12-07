import numpy as np
from metapy.algorithms.cat_swarm_optimization import Cat, CatSwarmOptimization


def rastringin_fitness(cat):
    return 10 * len(cat.position) + np.sum(np.square(cat.position) - 10 * np.cos(2*np.pi*cat.position))



class RastriginCSO(CatSwarmOptimization):
    def fitness(self, cat):
        return rastringin_fitness(cat)


cso = RastriginCSO(0.2, 20, [1, 1], 1, True,
                    2, [0.2, 0.2],
                    80, 0, True)
cso.init_population(10)


currentcats = cso.optimize(500)
print("Best cat evaluates to: " + str(min(currentcats)))
