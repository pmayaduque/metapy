from random import random
from copy import copy

import numpy as np


class CatSwarmOptimization:

    def __init__(self,
                 mr, smp, srd, cdc, spc,
                 c,
                 fitness, fitmax, fitmin, minimize=True):
        """
        Initialization for a cat swarm optimizer
        :param mr: Mixture Ratio, which percentage of the cat population should be in tracing mode
        :param smp: Seeking Memory Pool, number of cat-copies in seeking mode (per seeking cat)
        :param srd: Seeking Range per Dimension, how far should the cat be looking into the solution space
                    (m-dimensional vector)
        :param cdc: Counts of Dimensions to Change, how many dimensions to mutate in seeking mode
        :param spc: Self Position Consideration, can the cats own postion be a candidate for cat-copies in seeking mode?
        :param c: Constant velocity factor in tracing mode
        :param fitness: Function that should be used to evaluate the fitness of a cat (solution)
        :param fitmax: Max Value of the fitness function
        :param fitmin: Min Vlaue of the fitness function
        :param minimize: Default set to true, optimizer maximizes fitness when set to false
        """
        self.mr = mr
        self.smp = smp
        self.srd = srd
        self.cdc = cdc
        self.spc = spc
        self.c = c
        self.fitness = fitness
        self.fitmax = fitmax
        self.fitmin = fitmin
        self.minimize = minimize
        self._population = []
        self.bestcat = None

    def init_population(self, cats):
        """
        Copies a list of cats into a local initial population.
        Any present population is being deleted.
        :returns: length of copied population
        """
        self._population.clear()
        bestcat = cats[0]
        for c in cats:
            self._population.append(copy(c))
            if self.minimize:
                bestcat = c if c.eval() < bestcat.eval() else bestcat
            else:
                bestcat = c if c.eval() > bestcat.eval() else bestcat
        self.bestcat = bestcat
        return len(self._population)

    def optimize(self, epochs=1):
        #ToDo: currently no optimization is happening here .. this has to be fixed
        """
        Processes the tracing / seeking procedure for each cat once and evaluates their fitness afterwards.
        :param epochs: Number of iterations
        :return: List of fitness values for each cat in the population
        """
        print("Optimize for {} epochs\n .".format(epochs))
        for i in range(epochs):
            swap_cats = []
            for cat in self._population:
                if cat.is_seeking() is True:
                    # make smp copies of this cat
                    local_cats = []
                    # in case spc is true, there should only be smp-1 copies of the cat
                    nseekingcats = self.smp if self.spc is False else self.smp - 1
                    for i in range(nseekingcats):
                        local_cats.append(copy(cat))
                    # alter the position of all local cat-copies
                    for localcat in local_cats:
                        localcat.alter_position(self.srd)
                    # in case spc was true, the original cat has to be added now to the local_cats list
                    if self.spc is True:
                        local_cats.append(cat)
                    # Afterwards evaluate the fitness of all cats in that memory pool
                    local_fitness = []
                    for localcat in local_cats:
                        local_fitness.append(localcat.eval())
                    # select a new cat based on their respective fitness
                    local_probability = []
                    if self.minimize:
                        local_probability = [abs(fitness - self.fitmin) / abs(self.fitmax - self.fitmin)
                                             for fitness in local_fitness]
                    else:
                        local_probability = [abs(fitness - self.fitmax) / abs(self.fitmax - self.fitmin)
                                             for fitness in local_fitness]
                    norm_prob = local_probability / np.array(local_probability).sum()
                    swap_cats.append((cat, np.random.choice(local_cats, p=norm_prob)))
                else:  # cat has to be in tracing mode when not in seeking mode
                    cat.move(self.bestcat, self.c)
            for oldcat, newcat in swap_cats:
                self._population.remove(oldcat)
                self._population.append(newcat)
            for cat in self._population:
                if self.minimize:
                    self.bestcat = cat if cat.eval() < self.bestcat.eval() else self.bestcat
                else:
                    self.bestcat = cat if cat.eval() > self.bestcat.eval() else self.bestcat
            print(".", end="")

        print("\n DONE OPTIMIZING FOR {} epochs".format(epochs))
        return [cat.eval() for cat in self._population]


class Cat:

    def __init__(self, position, velocity, mr, fitness):
        """
        Initializes a Cat, that represents a possible solution in M-Dimensional solution-space
        :param position: M-Dimensional Vector in solution-space
        :param velocity: M-Dimensional Vector, indicating cats movement speed per dimension of solution-space
        :param mr: Mixture Ratio, percentage of cats in tracing mode [0.0, 1.0]
        :param fitness: Ref to fitness-function used to evaluate each single cat
        """
        self.position = np.array(position)
        self.velocity = np.array(velocity)
        self.fitness = fitness
        self._seeking = False if random() <= mr else True

    def eval(self):
        """
        Evaluate the cats fitness
        :returns: a fitness value for that cat
        """
        return self.fitness(self)

    @property
    def eval_prop(self):
        """
        Evaluate the cats fitness
        :returns: a fitness value for that cat
        """
        return self.fitness(self)

    def is_seeking(self):
        """
        Indicates whether this cat is in seeking mode
        :return: True - when cat is in seeking mode, False otherwise
        """
        return self._seeking

    def is_tracing(self):
        """
        Indicates whether this cat is in tracing mode
        :return: True - when cat is in tracing mode, False otherwise
        """
        return not self.is_seeking()

    def alter_position(self, srd):
        """
        Alters position of the cat based on (Majumder and Eldho 2016) Xcn = (1 + srd * random) * Xc
        This method is used for cats in seeking mode
        :param srd: Seeking Range per Dimension, how far should the cat be looking into the solution space
                    (m-dimensional vector)
        :returns: The new position of the cat
        """
        self.position = (1 + np.array(srd) * (random() - 0.5)) * self.position
        return self.position

    def move(self, bestcat, c):
        """
        This method moves cats in tracing mode into the direction of the currently best cat
        :param bestcat: Cat with highest fitness
        :param c: Constant factor for movement towards the best cats position
        :return: The new position of the cat
        """
        #ToDo: Normally this should be
        #       Velocity = Velocity + (newValue)
        # however this lead to pretty big velocities so had to change that
        self.velocity = random() * c * (bestcat.position - self.position)
        self.position = self.position + self.velocity
        return self.position
