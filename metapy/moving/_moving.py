from copy import copy
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


def alter_position_binary(cat, probability_of_mutation, counts_of_dimenstions_to_change):
    """
    Implementation follows (Sharafi Y.,Khanesar M.A., Teshnehlab M.:
    Discrete Binary Cat Swarm Optimization Algorithm)
    This method is used for cats in seeking mode.
    """
    new_cat = copy(cat)
    switch = np.random.choice(np.arange(len(cat)), counts_of_dimenstions_to_change, replace=False)
    for i in switch:
        should_switch = np.random.choice([True, False], 1, [probability_of_mutation,
                                                            1-probability_of_mutation])
        if should_switch[0] == True:
            new_cat[i] = 1 if new_cat[i] == 0 else 0
    return new_cat


def move_continuous(cat, bestcat, current_velocity, velocity_factor, max_velocity):
    """Following Chu and Tsai (probably, have to check this) a cat in tracing mode should
    calc their new velocity as vnew = nold + r * c (bestcat.pos - thiscat.pos).
    Velocity is limited by max_velocity, though.
    Args:
        cat(Vector): Current cats position
        bestcat(Vector): position of cat with best fitness
        current_velocity(Vector): velocity vector if current cat
        velocity_factor(float):
        max_velocity(float): maximum velocity

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


def move_binary(cat, bestcat, current_velocity, velocity_factor, max_velocity):
    """Following Chu and Tsai (probably, have to check this) a cat in tracing mode should
    calc their new velocity as vnew = nold + r * c (bestcat.pos - thiscat.pos).
    Velocity is limited by max_velocity, though.
    Args:
        cat(Vector): Current cat
        bestcat(Vector): cat with best fitness
        current_velocity(Vector): 2 Vectors,
                                    first representing probability of a dimension becoming 1
                                    second representing probability of a dimension becoming 0
        velocity_factor(float):
        max_velocity(Vector): Same as for 'current_velocity'

    Returns:
        new_cat(Vector): array of position of new cat
        velocity(Vector): array with new velocity

    ToDo: Currently inertia weights are randomly drawn from a uniform distribution.
          Should we make the inertia weights a new parameter to allow greater flexibility?
    """
    # draw random inertia_weight
    inertia_weight = np.random.random_sample()
    # calc new velocity values
    mask = list(map(lambda x: x if x == 1 else -1, bestcat))
    d1 = np.random.random_sample() * velocity_factor * np.array(mask)
    d0 = d1 * -1
    current_velocity[0] = inertia_weight * current_velocity[0] + d0
    current_velocity[1] = inertia_weight * current_velocity[1] + d1
    # limit new velocity to max_velocity
    v0 = list(map(lambda minVal, maxVal, vec: min(maxVal, max(minVal, vec)),
                  [max_velocity[0]] * len(current_velocity[0]),
                  [max_velocity[1]] * len(current_velocity[0]),
                  current_velocity[0]))
    v1 = list(map(lambda minVal, maxVal, vec: min(maxVal, max(minVal, vec)),
                  [max_velocity[0]] * len(current_velocity[1]),
                  [max_velocity[1]] * len(current_velocity[1]),
                  current_velocity[1]))
    v = list(map(lambda x, y, z: x if z == 0 else y, v1, v0, bestcat))
    # calc the probability of mutation based on new velocity
    t = 1 / (1 + np.exp(-1*np.array(v)))
    # switch the cats 0s or 1s based on probability vector t
    new_cat = copy(cat)
    for i in range(len(new_cat)):
        should_switch = np.random.choice([True, False], 1, [t[i], 1-t[i]])
        if should_switch[0] == True:
            new_cat[i] = bestcat[i]
    # return stuff
    return new_cat, [v0, v1]
