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
        Vector: new candidate
    """

    if extrapolate:
        u = np.random.uniform(-1, 2)
    else:
        u = np.random.uniform()
    new_candidate = u * candidates[0] + (1 - u) * candidates[1]
    return new_candidate

def order_based_crossover(candidates):
    """Performs a crossover which conserves the order of genes in the candidates. 12345678 + 26371485 -> **3456** + 2**71*8* -> 27345618

    Expects candidates to be same length, each gene appears in the candidates exactly once.
    
    Args:
        candidates (List(Vector)): two candidates
    
    Returns:
        Vector: new candidate
    """

    first, second = candidates[:2]
    assert len(first) == len(second)
    new_candidate = [None for i in range(len(first))]
    start = np.random.randint(0, len(first))
    end = np.random.randint(0, len(first))

    if start < end:
        for i in range(start, end):
            new_candidate[i] = first[i]
    else:
        for i in range(start, end, -1):
            new_candidate[i] = first[i]
    
    second = (gene for gene in second if gene not in new_candidate)
    for i in range(len(new_candidate)):
        if new_candidate[i] is None:
            new_candidate[i] = next(second)

    return np.array(new_candidate)
