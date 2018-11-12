import numpy as np
import random


def uniform_crossover(candidates):
    """Uniformly selects genotypes from the candidates. Expects candidates to have the same size.
    
    Args:
        candidates (List[Vectors]): two or more candidates
    
    Returns:
        Vector: new candidate
    """
    assert len(candidates) > 1
    length_of_new_candidate = len(random.choice(candidates))
    new_candidate = np.zeros(length_of_new_candidate)
    for i, genotypes in enumerate(zip(*candidates)):
        new_candidate[i] = random.choice(genotypes)
    return new_candidate

def one_point_crossover(candidates):
    """Performs one point crossover. Randomly chooses intersection point and takes first candidates genotypes up to 
    the intersection point and second candidates genes from intersection point to the end.
    
    Args:
        candidates (List(Vector)): two candidates
    
    Returns:
        Vector: new candidate
    """

    assert len(candidates) == 2
    intersection_point = np.random.randint(0, len(candidates[0]))
    new_candidate = np.hstack((candidates[0][:intersection_point], candidates[1][intersection_point:]))
    return new_candidate

def arithmetic_crossover(candidates, extrapolate=False):
    """TODO Add Docstring here
    
    Args:
        candidates ([type]): [description]
        extrapolate (bool, optional): Defaults to False. [description]
    
    Returns:
        [type]: [description]
    """

    if extrapolate:
        u = np.random.uniform(-1, 2)
    else:
        u = np.random.uniform()
    new_candidate = u * candidates[0] + (1 - u) * candidates[1]
    return new_candidate