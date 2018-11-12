import pytest
import numpy as np
import itertools
from metapy.selection import sample_from_fittest, rank_based_selection


def test_sample_from_fittest():
    population = [list(np.random.randint(0, 2, 10)) for i in range(20)]
    fitness = [i for i in range(20)]
    parents = sample_from_fittest(population, fitness, 5, 0.1)

    assert len(parents) == 5
    assert len(parents[0]) == 2

    selected_chromosomes = itertools.chain.from_iterable(parents)
    selected_chromosomes = [list(s) for s in selected_chromosomes]
    for unselected in population[2:]:
        assert unselected not in selected_chromosomes

    parents = sample_from_fittest(population, fitness, 5, 0.1, reverse=True)
    selected_chromosomes = itertools.chain.from_iterable(parents)
    selected_chromosomes = [list(s) for s in selected_chromosomes]
    for unselected in population[:-2]:
        assert unselected not in selected_chromosomes


def test_rank_based_selection():
    population = [list(np.random.randint(0, 2, 10)) for i in range(20)]
    fitness = [i for i in range(20)]
    parents = rank_based_selection(population, fitness, 5)
    # TODO how to test for random distributions?

    assert len(parents) == 5
