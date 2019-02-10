from copy import copy
from metapy.algorithms._base import Algorithm, Result

import numpy as np


class Cat(object):
    def __init__(self, position=None, velocity=None, mode=None):
        self.position = position
        self.velocity = velocity
        self.mode = mode


class CatSwarmOptimization(Algorithm):
    """
        Implementation of Cat Swarm Optimization Algorithm
        Args:
            seeking_range_per_dimension(Vector): How far should the cat be looking into the solution
                                            space (m-dimensional vector)
            max_velocity(Vector or float): Max Velocity for a cat in tracing mode
            seeking_memory_pool(int): Specifies the number of cat-copies in seeking mode
                                    (per seeking cat)
            count_of_dimensions_to_change(int): How many dimensions to mutate in seeking mode
            mixture_ratio(float): Indicates hich percentage of the cat population should be in
                              tracing mode
            self_position_consideration(bool): Can the cats own postion be a candidate for cat-copies
                                            in seeking mode?
            velocity_factor(float): Factor for velocity calculation in tracing mode
            fitmax(float): Max Value of the fitness function
            fitmin(float): Min Vlaue of the fitness function
            population_size(int): size of
            minimize(bool): Default set to true, optimizer maximizes fitness when set to false
        """

    def __init__(self, seeking_range_per_dimension, max_velocity, seeking_memory_pool,
                 count_of_dimensions_to_change, mixture_ratio=0.2, self_position_consideration=True,
                 velocity_factor=2, fitmax=1e6, fitmin=0.0, population_size=10, minimize=True):
        Algorithm.__init__(self)
        self.mixture_ratio = mixture_ratio
        self.population_size = population_size
        self.seeking_memory_pool = seeking_memory_pool
        self.seeking_range_per_dimension = seeking_range_per_dimension
        self.count_of_dimensions_to_change = count_of_dimensions_to_change
        self.self_position_consideration = self_position_consideration
        self.velocity_factor = velocity_factor
        self.max_velocity = max_velocity
        self.fitmax = fitmax
        self.fitmin = fitmin
        self.minimize = minimize

        self.initial_population = None
        self._population = None
        self.bestcat = None

        self.res = Result()

    def optimize(self, max_iterations=np.inf, max_time=np.inf):
        """
        Processes the tracing / seeking procedure for each cat. Stop after 'max_iterations' or
        'max_time' whatever is first. It is sufficient to define one of those, though.
        In the process the actual fitness values used for calculation of propability of selection
        are limited to the min and max fitness values given at the time of initialization.
        This has no effect when the actual min and max fitness values are well known,
        however inaccurate estimates would have a chatastrophic effect on the selection process.
        Running this function multiple times reinitializes the population before running the
        optimization again.
        Args:
            max_iterations(int): Number of iterations for which the algorithm should stop optimizing
            max_time(float): Realtime after which the algorithm should stop optimizing
        Returns:
            Result: optimizer result object containing solution, metadata and progress information
        """
        self.max_iterations = max_iterations
        self.max_time = max_time
        if np.isinf(self.max_iterations) and np.isinf(self.max_time):
            raise TypeError('Expect either max_iterations or max_time to be not inf.')

        self._init_result()
        self._init_population()
        self.start()
        while not self.has_finished():
            swap_cats = []
            for cat in self._population:
                if self.has_finished():
                    break
                if cat.mode == "tracing":
                    self._move(cat, self.bestcat)
                    continue
                # Seeking mode here
                # in case self_position_consideration is true,
                # there should only be seeking_memory_pool-1 copies of the cat
                nseekingcats = self.seeking_memory_pool if not self.self_position_consideration \
                    else self.seeking_memory_pool - 1

                # make nseekingcats copies of this cat
                local_cats = [copy(cat)] * nseekingcats
                # alter their position
                local_cats = [self._alter_position(c) for c in local_cats]

                # in case self_position_consideration was true,
                # the original cat has to be added now to local_cats
                if self.self_position_consideration:
                    local_cats.append(cat)

                # Afterwards evaluate the fitness of all cats in that memory pool
                local_fitness = [self._fitness(c) for c in local_cats]

                # select a new cat based on their respective fitness
                if self.minimize:
                    local_fitness = [min(fit, self.fitmax)
                                     for fit in local_fitness]
                    local_probability = [(self.fitmax - fitness) / abs(self.fitmax - self.fitmin)
                                         for fitness in local_fitness]
                else:
                    local_fitness = [max(fit, self.fitmin)
                                     for fit in local_fitness]
                    local_probability = [abs(fitness - abs(self.fitmin)) /
                                         abs(self.fitmax - self.fitmin)
                                         for fitness in local_fitness]

                norm_prob = local_probability / np.array(local_probability).sum()
                swap_cats.append((cat, np.random.choice(local_cats, p=norm_prob)))

            for oldcat, newcat in swap_cats:
                self._population.remove(oldcat)
                self._population.append(newcat)

            # get best cat currently in population
            def key(c): return self._fitness(c)
            self.bestcat = min(self._population, key=key) if self.minimize else max(
                self._population, key=key)

            self.res.best_progress.append(self._fitness(self.bestcat))
            self.res.averaged_progress.append(
                sum([self._fitness(c) for c in self._population]) / len(self._population))
            self.iterations += 1

        self.res.solution = self.bestcat.position
        return self.res

    def init_population(self):
        """
        Generate positions for initial population
        Returns:
            A list of positions for the initial population
        """
        raise NotImplementedError

    def init_velocities(self):
        """
        Generate velocities for the initiali population.
        The generated velocity list has to be of the same length as the positions-list.
        Returns:
            A list of velocities for the initial population
        """
        raise NotImplementedError

    def _init_population(self):
        self.initial_population = self.init_population()
        self._population = []
        velocities = self.init_velocities()
        for i, position in enumerate(self.initial_population):
            mode = "tracing" if np.random.uniform() < self.mixture_ratio else "seeking"
            self._population.append(Cat(position=position, velocity=velocities[i], mode=mode))

        if self.minimize is True:
            def return_better(a, b): return a if self._fitness(a) < self._fitness(b) else b
        else:
            def return_better(a, b): return a if self._fitness(a) > self._fitness(b) else b
        self.bestcat = self._population[0]
        for cat in self._population:
            self.bestcat = return_better(cat, self.bestcat)

    def _init_result(self):
        self.res = Result()
        self.res.optimizer_settings['mixture_ratio'] = self.mixture_ratio
        self.res.optimizer_settings['seeking_memory_pool'] = self.seeking_memory_pool
        self.res.optimizer_settings['seeking_range_per_dimension'] = \
            self.seeking_range_per_dimension
        self.res.optimizer_settings['count_of_dimensions_to_change'] = \
            self.count_of_dimensions_to_change
        self.res.optimizer_settings['self_position_consideraiton'] = \
            self.self_position_consideration
        self.res.optimizer_settings['velocity_factor'] = self.velocity_factor
        self.res.optimizer_settings['max_velocity'] = self.max_velocity
        self.res.optimizer_settings['fitmax'] = self.fitmax
        self.res.optimizer_settings['fitmin'] = self.fitmin
        self.res.function_calls['fitness'] = 0
        self.res.function_calls['move'] = 0
        self.res.function_calls['alter_position'] = 0

    def _fitness(self, cat):
        self.res.function_calls['fitness'] += 1
        return self.fitness(cat.position)

    def fitness(self, cat):
        """
        Evaluates the fitness of the cat.
        Has to be implemented when inheriting this class
        Args:
            cat(Vector)
        Returns:
            float: fitness of cat, value of objective function
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
        self.res.function_calls['move'] += 1
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
            cat(Cat): cat with position, velocity and mode
            seeking_range_per_dimension (Vector): how far should the cat be looking into the solution space
            count_of_dimensions_to_change (Int): how many dimensions to mutate in seeking mode

        Returns:
            Cat
        """
        self.res.function_calls['alter_position'] += 1
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


class BinaryCatSwarmOptimization(CatSwarmOptimization):
    """
        Implementation of Binary Cat Swarm Optimization Algorithm
        Args:
    
            mixture_ratio(float): Mixture Ratio, which percentage of the cat population
                              should be in tracing mode
            seeking_memory_pool(int): Seeking Memory Pool,
                                    number of cat-copies in seeking mode (per seeking cat)
            probability_mutation_operation(float): seeking mode is a binary mutation in
                                                   binary cat swarm optimization. So what once was
                                                   the seeking range per dimension has become the
                                                   probability of a mutation operation.
            count_of_dimensions_to_change(int): Counts of Dimensions to Change,
                                              meaning how many dimensions to mutate in seeking mode
            self_position_consideration(bool): Self Position Consideration, can the cats own position
                                            be a candidate for cat-copies in seeking mode?
            velocity_factor(float): Arbitrary float value
            max_velocity(Vector): 2D-Vector matrix according to velocity_factor
            fitmax(float): Max Value of the fitness function
            fitmin(float): Min Vlaue of the fitness function
            minimize(float): Default set to true, optimizer maximizes fitness when set to false
        """
    def __init__(self, mixture_ratio, seeking_memory_pool, probability_mutation_operation,
                 count_of_dimensions_to_change, self_position_consideration, velocity_factor,
                 max_velocity, fitmax, fitmin, population_size=None, minimize=True):

        CatSwarmOptimization.__init__(self, probability_mutation_operation, max_velocity,
                                      seeking_memory_pool, count_of_dimensions_to_change,
                                      mixture_ratio, self_position_consideration, velocity_factor,
                                      fitmax, fitmin, population_size, minimize)

    def alter_position(self, cat, probability_mutation_operation, count_of_dimensions_to_change):
        """Method for altering a cats position if its a tracing cat

        Args:
            cat (Vector): position of cat
            probability_mutation_operation (Vector): how far should the cat be looking into the
                                                     solution space
            count_of_dimensions_to_change (Int): how many dimensions to mutate in seeking mode

        Returns:
            cat (Vector): New position
        """
        raise NotImplementedError

    def _init_result(self):
        self.res = Result()
        self.res.optimizer_settings['mixture_ratio'] = self.mixture_ratio
        self.res.optimizer_settings['seeking_memory_pool'] = self.seeking_memory_pool
        self.res.optimizer_settings['probability_mutation_operation'] = \
            self.seeking_range_per_dimension
        self.res.optimizer_settings['count_of_dimensions_to_change'] = \
            self.count_of_dimensions_to_change
        self.res.optimizer_settings['self_position_consideraiton'] = \
            self.self_position_consideration
        self.res.optimizer_settings['velocity_factor'] = self.velocity_factor
        self.res.optimizer_settings['max_velocity'] = self.max_velocity
        self.res.optimizer_settings['fitmax'] = self.fitmax
        self.res.optimizer_settings['fitmin'] = self.fitmin
        self.res.function_calls['fitness'] = 0
        self.res.function_calls['move'] = 0
        self.res.function_calls['alter_position'] = 0
