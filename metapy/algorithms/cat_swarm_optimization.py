from random import random
from copy import copy
from collections import namedtuple

import numpy as np


class CatSwatmOptimizationResult(object):
    """This class holds the result of the CSO. It keeps the most important parameters as well as ongoing fitness values
    in order to get an idea of how the algorithm has performed
    """

    def __init__(self, cat_swarm_optimizer):
        self.mixture_ratio = cat_swarm_optimizer.mixture_ratio
        self.seeking_memory_pool = cat_swarm_optimizer.seeking_memory_pool
        self.seeking_range_per_dimension = cat_swarm_optimizer.seeking_range_per_dimension
        self.count_of_dimensions_to_change = cat_swarm_optimizer.count_of_dimensions_to_change
        self.self_position_consideration = cat_swarm_optimizer.self_position_consideration
        self.velocity_factor = cat_swarm_optimizer.velocity_factor
        self.max_velocity = cat_swarm_optimizer.max_velocity
        self.fitmax = cat_swarm_optimizer.fitmax
        self.fitmin = cat_swarm_optimizer.fitmin

        self.average_fitness = []
        self.best_fitness = []
        self.solution = None


class Cat(object):
    def __init__(self, position=None, velocity=None, mode=None):
        self.position = position
        self.velocity = velocity
        self.mode = mode


class CatSwarmOptimization(object):
    """
        Implementation of Cat Swarm Optimization Algorithm
        :param mixture_ratio: Mixture Ratio, which percentage of the cat population should be in tracing mode
        :param seeking_memory_pool: Seeking Memory Pool, number of cat-copies in seeking mode (per seeking cat)
        :param seeking_range_per_dimension: Seeking Range per Dimension, how far should the cat be looking into the solution
                    space (m-dimensional vector)
        :param count_of_dimensions_to_change: Counts of Dimensions to Change, how many dimensions to mutate in seeking mode
        :param self_position_consideration: Self Position Consideration, can the cats own postion be a candidate for
                    cat-copies in seeking mode?
        :param c: Constant velocity factor in tracing mode
        :param max_velocity: Max Velocity for a cat in tracing mode
        :param fitmax: Max Value of the fitness function
        :param fitmin: Min Vlaue of the fitness function
        :param minimize: Default set to true, optimizer maximizes fitness when set to false
        """

    def __init__(self, mixture_ratio, seeking_memory_pool, seeking_range_per_dimension,
                 count_of_dimensions_to_change, self_position_consideration, velocity_factor,
                 max_velocity, fitmax, fitmin, population_size=None, minimize=True):

        self.mixture_ratio = mixture_ratio
        self.seeking_memory_pool = seeking_memory_pool
        self.seeking_range_per_dimension = seeking_range_per_dimension
        self.count_of_dimensions_to_change = count_of_dimensions_to_change
        self.self_position_consideration = self_position_consideration
        self.velocity_factor = velocity_factor
        self.max_velocity = max_velocity
        self.fitmax = fitmax
        self.fitmin = fitmin
        self.minimize = minimize

        self.bestcat = None

        self.population_size = population_size
        self.initial_population = None

        self._population = []

    def optimize(self, epochs=1):
        """
        Processes the tracing / seeking procedure for each cat 'epoch' times  and evaluates
        their fitness.
        In the process the actual fitness values used for calculation of propability of selection
        are limited to the min and max fitness values given at the time of initialization.
        This has no effect when the actual min and max fitness values are well known,
        however inaccurate estimates would have a chatastrophic effect on the selection process.
        :param epochs: Number of iterations
        :return: List of fitness values for each cat in the final population.
                 Do note that the best final cat does not have to be the
                 overall best cat seen in cso!
        """
        if self.initial_population is None:
            self.init_population()

        for position in self.population:
            velocity = np.random.uniform(
                low=-0.5, high=0.5) * self.max_velocity
            mode = "tracing" if np.random.uniform() < self.mixture_ratio else "seeking"
            self._population.append(
                Cat(position=position, velocity=velocity, mode=mode))

        # get best cat currently in population
        def key(c): return self.fitness(c.position)
        self.bestcat = min(self._population, key=key) if self.minimize else max(
            self._population, key=key)

        res = CatSwatmOptimizationResult(self)

        epoch = 0
        while epoch < epochs:
            swap_cats = []
            for cat in self._population:
                if cat.mode == "tracing":
                    cat = self._move(cat, self.bestcat)
                    continue
                # Seeking mode here
                # in case self_position_consideration is true, there should only be seeking_memory_pool-1 copies of the cat
                nseekingcats = self.seeking_memory_pool if self.self_position_consideration is False else self.seeking_memory_pool - 1

                # make nseekingcats copies of this cat
                local_cats = [copy(cat) for i in range(nseekingcats)]
                # alter their position
                local_cats = [self._alter_position(c) for c in local_cats]

                # in case self_position_consideration was true, the original cat has to be added now to local_cats
                if self.self_position_consideration is True:
                    local_cats.append(cat)

                # Afterwards evaluate the fitness of all cats in that memory pool
                local_fitness = [self.fitness(c.position) for c in local_cats]

                # select a new cat based on their respective fitness
                if self.minimize:
                    local_fitness = [min(fit, self.fitmax)
                                     for fit in local_fitness]
                    local_probability = [abs(fitness - self.fitmax) / abs(self.fitmax - self.fitmin)
                                         for fitness in local_fitness]
                else:
                    local_fitness = [max(fit, self.fitmin)
                                     for fit in local_fitness]
                    local_probability = [fitness - self.fitmin / abs(self.fitmax - self.fitmin)
                                         for fitness in local_fitness]

                norm_prob = local_probability / \
                    np.array(local_probability).sum()
                swap_cats.append(
                    (cat, np.random.choice(local_cats, p=norm_prob)))

            for oldcat, newcat in swap_cats:
                self._population.remove(oldcat)
                self._population.append(newcat)

            # get best cat currently in population
            def key(c): return self.fitness(c.position)
            self.bestcat = min(self._population, key=key) if self.minimize else max(
                self._population, key=key)

            res.best_fitness.append(self.fitness(self.bestcat.position))
            res.average_fitness.append(
                sum([self.fitness(c.position) for c in self._population]) / len(self._population))
            epoch += 1

        res.solution = (self.fitness(self.bestcat.position),
                        self.bestcat.position)
        return res

    def init_population(self):
        """
        write candidate vectors into self.population
        """
        raise NotImplementedError

    def fitness(self, cat):
        """
        Evaluates the fitness of the cat.
        Has to be implemented when inheriting this class
        :returns: The fitness of the Cat
        """
        raise NotImplementedError

    def _move(self, cat, best_cat):
        """Internal method for calling move

        Args:
            cat (Cat): Current candidate cat, which performs move
            best_cat (Cat): Best cat currently
        Returns:
            Cat at new position with updated velocity
        """
        old_mode = cat.mode
        cat_position, velocity = self.move(
            cat.position, best_cat.position, cat.velocity, self.velocity_factor, self.max_velocity)
        return Cat(position=cat_position, velocity=velocity, mode=old_mode)

    def move(self, cat, best_cat, current_velocity, velocity_factor, max_velocity):
        """Method for moving one cat towards the best cat

        Args:
            cat (Vector): Current position of the cat
            best_cat (Vector): Position of best cat
            current_velocity (float/Vector): Velocity of the cat
            max_velocity (float/Vector): Max Velocity
        Returns:
            cat(Vector): position of cat after move
            velocity(floatVector): new velocity
        """
        raise NotImplementedError

    def _alter_position(self, cat):
        """Method for altering a cats position if a cat is tracing

        Args:
            cat (Vector): cat
            seeking_range_per_dimension (Vector): how far should the cat be looking into the solution space
            count_of_dimensions_to_change (Int): how many dimensions to mutate in seeking mode

        Returns:
            Cat
        """
        old_velocity = cat.velocity
        old_mode = cat.mode
        new_position = self.alter_position(
            cat.position, self.seeking_range_per_dimension, self.count_of_dimensions_to_change)
        return Cat(position=new_position, velocity=old_velocity, mode=old_mode)

    def alter_position(self, cat, seeking_range_per_dimension, count_of_dimensions_to_change):
        """Method for altering a cats position if its a tracing cat

        Args:
            cat (Vector): position of cat
            seeking_range_per_dimension (Vector): how far should the cat be looking into the solution space
            count_of_dimensions_to_change (Int): how many dimensions to mutate in seeking mode

        Returns:
            cat (Vector): New position
        """
        raise NotImplementedError
