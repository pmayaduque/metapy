from random import random

class CatSwarmOptimization:

    def __init__(self,
                 mr, smp, srd, cdc, spc,
                 c, fitness):
        """
        Initialization for a cat swarm optimizer
        :param maxvelo: maximum velocity of cats movement speed in any dimension
        :param mr: Mixture Ratio, which percentage of the cat population should be in tracing mode
        :param smp: Seeking Memory Pool, number of cat-copies in seeking mode (per seeking cat)
        :param srd: Seeking Range per Dimension, how far should the cat be looking into the solution space
        :param cdc: Counts of Dimensions to Change, how many dimensions to mutate in seeking mode
        :param spc: Self Position Consideration, can the cats own psotion be a candidate for cat-copies in seeking mode?
        :param c: Constant velocity factor in tracing mode
        :param fitness: Function that should be used to evaluate the fitness of a cat (solution)
        """
        self.mr = mr
        self.smp = smp
        self.srd = srd
        self.cdc = cdc
        self.spc = spc
        self.c = c
        self.fitness = fitness
        self._population = []

    def init_population(self, cats):
        """
        Copies a list of cats into a local initial population.
        Any present population is being deleted.
        :returns: length of copied population
        """
        self._population.clear()
        for c in cats:
            self._population.append(c.copy())
        return len(self._population)

    def iterate(self):
        """
        Processes the tracing / seeking procedure for each cat once and evaluates their fitness afterwards.
        :return: List of fitness values for each cat in the population
        """
        for cat in self._population:


class Cat:

    def __init__(self, position, velocity, mr, fitness):
        """
        Initializes a Cat, that represents a possible solution in M-Dimensional solution-space
        :param position: M-Dimensional Vector in solution-space
        :param velocity: M-Dimensional Vector, indicating cats movement speed per dimension of solution-space
        :param mr: Mixture Ratio, percentage of cats in tracing mode [0.0, 1.0]
        :param fitness: Ref to fitness-function used to evaluate each single cat
        """
        self.position = position
        self.velocity = velocity
        self.fitness = fitness
        self._seeking = True if random() <= mr else False

    def eval(self):
        """
        Evaluate the cats fitness
        :returns: a fitness value for that cat
        """
        return self.fitness(self)
