import numpy as np


class Distribution:
    def __init__(self, num_params, highs, lows):
        """ Computes parameters for the individuals of a environment class"""

        self.__num_params = num_params
        self.__highs = highs
        self.__lows = lows

        if len(self.__highs) != self.__num_params or len(self.__lows) != self.__num_params:
            raise ValueError("Both the length of the highs and lows array must agree with the number of parameters")

    def uniform(self, param_i, number_of):
        return np.uniform(self.__lows[param_i], self.__highs[param_i], number_of)

    def get_high(self, param_i):
        return self.__highs[param_i]

    def get_low(self, param_i):
        return self.__lows[param_i]