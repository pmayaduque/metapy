import numpy as np


class SimulatedAnnealingOptimizationResult(object):
    def __init__(self, temperature, annealing_factor):
        self.initial_temperature = temperature
        self.annealing_factor = annealing_factor
        self.state = None
        self.energies = []

    def __str__(self):
        d = {'energies': self.energies,
             'state': self.state,
             'annealing_factor': self.annealing_factor,
             'initial temperature': self.initial_temperature}
        return str(d)


class SimulatedAnnealing(object):

    def __init__(self, state=None, init_temperature=None, minimize=True):
        self.state = state
        self.temperature = init_temperature
        self.current_energy = None

        self.minimize = minimize

    def optimize(self, max_iterations=2000, annealing_factor=0.999):

        if self.state is None:
            self.init_state()

        self.current_energy = self.energy(self.state)

        result = SimulatedAnnealingOptimizationResult(self.temperature, annealing_factor)
        result.energies.append(self.current_energy)

        for i in range(max_iterations):
            
            candidate = self.alter(self.state)
            candidate_energy = self.energy(candidate)

            if self.is_better(candidate_energy, self.current_energy):
                self.state = candidate
                self.current_energy = candidate_energy
            else:
                probability = np.exp(-np.abs(self.current_energy - candidate_energy) / (self.temperature))
                if np.random.uniform() < probability:
                    self.state = candidate
                    self.current_energy = candidate_energy

            result.energies.append(self.current_energy)
            self.temperature *= annealing_factor

        result.state = self.state
        return result

    def energy(self, state):
        """The energy function is the objective function (or cost function). Needs to be provided by the user.
        
        Args:
            state (Vector)
        Returns:
            energy: float
        """

        raise NotImplementedError

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
