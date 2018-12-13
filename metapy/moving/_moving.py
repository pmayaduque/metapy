import numpy as np
from random import random


def alter_position_continuous(cat, seeking_range_per_direction, counts_of_dimenstions_to_change):
    """
    Following (Majumder and Eldho 2016) the position should be altered as
    Xcn = (1 + seeking_range_per_direction * r) * Xc, where r is a random number between 0 and 1.
    However going with a random number between -0.5 and 0.5 lead to better results on the
    rastrigin and rosenbrock function, so we deviated slightly here.
    This method is used for cats in seeking mode.
    """
    mask = np.zeros(len(seeking_range_per_direction))
    switch = np.random.choice(np.arange(len(seeking_range_per_direction)), counts_of_dimenstions_to_change, replace=False)
    for i in switch:
        mask[i] = 1
    new_cat = (1 + (np.array(seeking_range_per_direction) * (random() - 0.5) * mask)) * cat
    return new_cat

def move_continuous(cat, bestcat, current_velocity, velocity_factor, max_velocity):
    """Following Chu and Tsai (probably, have to check this) a cat in tracing mode should
    calc their new velocity as vnew = nold + r * c (bestcat.pos - thiscat.pos).
    Velocity is limited by max_velocity, though.
    Args:
        cat(Vector): Current cat
        bestcat(Vector): cat with best fitness
        current_velocity(Vector): velocity vector if current cat
        velocity_factor(float):
        max_velocity()

    Returns:
        new_cat(Vector): array of position of new cat
        velocity(Vector): array with new velocity
    """

    # calc new velo
    velocity = current_velocity + np.random.uniform(low=-1, high=1) * velocity_factor * (bestcat - cat)
    # limit velocity
    velocity = np.clip(velocity, -max_velocity, max_velocity)
    # update position
    new_cat = cat + velocity
    return new_cat, velocity
