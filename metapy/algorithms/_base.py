import numpy as np
import time


class Algorithm(object):
    """Base class for all algorithms, which holds exit conditions.
    """
    def __init__(self):
        self.max_iterations = np.inf
        self.max_time = np.inf
        self.iterations = 0
        self.start_time = None
        self.external_exit = False
        # ToDo: We shouldnt be using those, but instead use the results function_calls property
        self.function_calls = {}
    
    def start(self):
        self.start_time = time.time()
    
    def has_finished(self):
        time = self.elapsed_time > self.max_time
        iterations = self.iterations > self.max_iterations
        return time or iterations or self.external_exit
    
    @property
    def elapsed_time(self):
        return time.time() - self.start_time


class Result(object):
    """This Result class should either be used directly as the algorithm output / result
    or the prefered result should inherit from this class
    """
    def __init__(self):
        self.solution = None
        self.averaged_progress = []
        self.best_progress = []
        self.optimizer_settings = {}
        self.function_calls = {}
