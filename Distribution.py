import numpy as np


class Distribution:
    def __init__(self, highs, lows):
        """ Computes parameters for the individuals of a environment class"""

        self.__highs = highs
        self.__lows = lows

        if len(self.__highs) != len(self.__lows):
            raise ValueError("Number of highs must match number of lows")

        self.num_params = len(self.__highs)

    def uniform(self, param_i, number_of):
        return np.random.uniform(self.__lows[param_i], self.__highs[param_i], number_of)

    def get_high(self, param_i):
        return self.__highs[param_i]

    def get_low(self, param_i):
        return self.__lows[param_i]

    def range(self, param_i):
        return self.__highs[param_i] - self.__lows[param_i]

    def __str__(self):
        as_string = ""
        for idx in range(self.num_params):
            as_string += f"{self.__lows[idx]} {self.__highs[idx]} "

        return as_string