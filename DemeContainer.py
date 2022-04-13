import numpy as np
from Individual import Individual
from Deme import Deme


class DemeContainer(object):
    def __init__(self, cross_over_type: str, fitness_type: str, num_demes: int):
        self.num_demes = num_demes
        self.demes = [[] for _ in range(self.num_demes)]
        self.cross_over_type = cross_over_type
        self.fitness_type = fitness_type

    def append(self, deme: Deme):
        self.demes = np.append(self.demes, deme)

    def __getitem__(self, i):
        return self.demes[i]

    def __setitem__(self, i, value):
        self.demes[i] = value

    def genetic_drift(self, deme_i: int, final_t: int):
        return 0, 0

    def __get__(self, i: int):
        return self.demes[i]


class RingTopology(DemeContainer):
    def __init__(self, cross_over_type: str, fitness_type: str, num_demes: int):
        super().__init__(cross_over_type, fitness_type, num_demes)

    def genetic_drift(self, deme_i: int, final_t: int):
        left_neighbour = self.demes[self.num_demes - 1] if deme_i == 0 else self.demes[deme_i - 1]
        right_neighbour = self.demes[0] if deme_i == self.num_demes - 1 else self.demes[deme_i + 1]

        chosen_deme = [left_neighbour, right_neighbour][np.random.randint(0, 2)]
        i = np.random.randint(0, chosen_deme.pop_size - 1)
        individual = chosen_deme.individuals[i]

        chosen_deme.individuals[i] = self.demes[deme_i].individuals[-1].cross_over(individual, self.cross_over_type, final_t)
        chosen_deme.sink(i, final_t, self.fitness_type)

        return chosen_deme, i

