import numpy as np


def fix_fitness_plot(fitness_values: np.array):
    """ Alter fitness array to ensure fitness plots look like step plot"""

    generations = []
    new_values = []
    for i in range(0, len(fitness_values)-1):
        curr_v = fitness_values[i]
        next_v = fitness_values[i + 1]

        generations.append(i)
        new_values.append(curr_v)

        if abs(curr_v - next_v) > 0.01:
            generations.append(i)
            new_values.append(next_v)

    return generations, new_values