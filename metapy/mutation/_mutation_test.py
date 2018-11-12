import pytest
import numpy as np
from metapy.mutation import bit_flip_mutation, swap_mutation, rand_int_mutation


def test_bit_flip_mutation():
    candidate = np.random.randint(0, 2, 10)
    mutated_candidate = bit_flip_mutation(candidate)

    # count unequal bits
    count = 0
    for a, b in zip(candidate, mutated_candidate):
        if a != b:
            count += 1
    
    assert count == 1

def test_swap_mutation():
    candidate = np.random.randint(0, 100, 10)
    mutated_candidate = swap_mutation(candidate)

    # count unequal bits
    count = 0
    for a, b in zip(candidate, mutated_candidate):
        if a != b:
            count += 1
    
    assert count == 2

    assert len(set(candidate)) == len(set(mutated_candidate))

def test_rand_int_mutation():
    candidate = np.random.randint(0, 2, 10)
    mutated_candidate = rand_int_mutation(candidate, 3, 4)
    assert 3 in mutated_candidate
