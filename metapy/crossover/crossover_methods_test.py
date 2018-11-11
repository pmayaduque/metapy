import pytest
import numpy as np
from metapy.crossover import uniform_crossover, one_point_crossover


def test_uniform_crossover():
    candidates = [np.random.randint(0, 10, size=10) for i in range(3)]
    new_candidate = uniform_crossover(candidates)

    for i in range(len(new_candidate)):
        genotype = new_candidate[i]
        possible_genotypes = [c[i] for c in candidates]
        assert genotype in possible_genotypes

def test_one_point_crossover():
    candidates = [np.random.randint(0, 10, size=10) for i in range(2)]
    new_candidate = one_point_crossover(candidates)

    for i in range(len(new_candidate)):
        genotype = new_candidate[i]
        possible_genotypes = [c[i] for c in candidates]
        assert genotype in possible_genotypes
