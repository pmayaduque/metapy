import numpy as np


def bit_flip_mutation(candidate):
    """Randomly selects one position and flips the bit.

    Args:
        candidate (Vector): chromosome

    Returns:
        Vector: chromosome
    """

    bit_to_rotate = np.random.randint(0, len(candidate))
    new_candidate = np.copy(candidate)
    new_candidate[bit_to_rotate] = 0 if new_candidate[bit_to_rotate] == 1 else 1
    return new_candidate


def rand_int_mutation(candidate, low=0, high=1):
    """Inserts a random integer between low and high at a random position

    Args:
        candidate (Vector): chromosome
        low (int, optional): Defaults to 0.
        high (int, optional): Defaults to 1.
    """
    pos = np.random.randint(0, len(candidate))
    new_candidate = np.copy(candidate)
    new_candidate[pos] = np.random.randint(low, high)
    return new_candidate


def swap_mutation(candidate):
    """Take two positions on the chromosome and interchange these values.

    Args:
        candidate (Vector): chromosome

    Returns:
        candidate (Vector): chromosome
    """
    idx_a, idx_b = np.random.randint(0, len(candidate), 2)
    itm_a, itm_b = candidate[idx_a], candidate[idx_b]
    new_candidate = np.copy(candidate)
    new_candidate[idx_a] = itm_b
    new_candidate[idx_b] = itm_a
    return new_candidate
