import numpy as np
import time
from metapy.algorithms._base import Algorithm, Result


class SimulatedAnnealing(Algorithm):

    def __init__(self, state=None, init_temperature=None, minimize=True):
        Algorithm.__init__(self)
        self.state = state
        self.temperature = init_temperature
        self.current_energy = None

        self.minimize = minimize
        self.function_calls = {'energy': 0, 'alter': 0, 'state changes': 0}

    def optimize(self, max_iterations=np.inf, max_time=np.inf, annealing_factor=None):
        
        self.max_iterations = max_iterations
        self.max_time = max_time
        
        if np.isinf(self.max_iterations) and np.isinf(self.max_time):
            raise TypeError('Expect either max_iterations or max_time to be not inf.')
        
        if self.state is None:
            self.init_state()
        
        # estimate number of iterations by measuring time for objective function
        if not np.isinf(self.max_time):
            start_time = time.time()
            self.current_energy = self._energy(self.state)
            time_needed = time.time() - start_time
            estimated_iterations = int(self.max_time / time_needed)
        else:
            self.current_energy = self._energy(self.state)
        
        # make educated guess for good initial temperature
        if self.temperature is None:
            if not np.isinf(self.max_iterations):
                self.temperature = self.max_iterations / 50
            else:
                self.temperature = estimated_iterations / 50
        
        # make educated guess for good annealing factor
        if annealing_factor is None:
            goal_temperature = 1e-1
            if not np.isinf(self.max_iterations):                
                annealing_factor = np.power(goal_temperature / self.temperature, 1 / self.max_iterations)
            else:
                annealing_factor = np.power(goal_temperature / self.temperature, 1 / estimated_iterations)


        res = Result()
        res.optimizer_settings = {
            'initial_temperature': self.temperature,
            'annealing_factor': annealing_factor,
            'max_iterations': self.max_iterations,
            'max_time': self.max_time}

        res.best_progress.append(self.current_energy)
        
        self.start()
        while not self.has_finished():
            
            candidate = self._alter(self.state)
            candidate_energy = self._energy(candidate)

            if self.is_better(candidate_energy, self.current_energy):
                self.state = candidate
                self.current_energy = candidate_energy
                self.function_calls['state changes'] += 1
            else:
                probability = np.exp(-np.abs(self.current_energy - candidate_energy) / (self.temperature))
                if np.random.uniform() < probability:
                    self.state = candidate
                    self.current_energy = candidate_energy
                    self.function_calls['state changes'] += 1

            res.best_progress.append(self.current_energy)
            self.temperature *= annealing_factor
            self.iterations += 1

        res.solution = self.state
        res.function_calls = self.function_calls
        return res
    
    def _energy(self, state):
        self.function_calls['energy'] += 1
        return self.energy(state)

    def energy(self, state):
        """The energy function is the objective function (or cost function). Needs to be provided by the user.
        
        Args:
            state (Vector)
        Returns:
            energy: float
        """

        raise NotImplementedError
    
    def _alter(self, state):
        self.function_calls['alter'] += 1
        return self.alter(state)

    def alter(self, state):
        """Function to alter the current state. Should deliver "neighbors" of current state.
        
        Args:
            state (Vector)
        Returns:
            new_state (Vector)
        """

        raise NotImplementedError

    def init_state(self):
        """In case no initial state is provided, this method is used to generate an initial guess
        """
        raise NotImplementedError

    def is_better(self, energy_a, energy_b):
        """True, if energy_a is better or equal to energy_b"""
        if self.minimize:
            return energy_a <= energy_b
        else:
            return energy_a >= energy_b
