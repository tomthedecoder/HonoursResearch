import numpy as np
from dataclasses import dataclass, field


@dataclass(unsafe_hash=True)
class Distribution:

    parameters: list = field(init=True)

    def __post_init__(self):
        self.num_parameters = len(self.parameters)

    def sample_weights(self, number_of, connection_array):
        pass

    def sample_other(self, param_i, number_of):
        pass

    def get_type(self):
        return "Virtual"


class Uniform(Distribution):

    def __post_init__(self):
        self.lows = self.parameters[0]
        self.highs = self.parameters[1]

    def sample_weights(self, number_of, connection_array):
        weights = np.random.uniform(self.lows[0] if -0.02 < self.lows[0] < 0.02 else [0.1, -0.1][np.random.randint(0, 2)], self.highs[0], number_of)
        return np.array([10 * weights[i-1] if i == j else weights[i-1] for i, j in connection_array])

    def sample_other(self, param_i, number_of):
        return np.random.uniform(self.lows[param_i], self.highs[param_i], number_of)

    def range(self, param_i):
        return self.highs[param_i] - self.lows[param_i]

    def __str__(self):
        as_string = ""
        for idx, low_value in enumerate(self.lows):
            high_value = self.highs[idx]
            as_string += f"{low_value},{high_value} "

        return as_string

    def get_type(self):
        return "Uniform"


class Poisson(Distribution):

    def __post_init__(self):
        self.lambdas = self.parameters

    def sample_weights(self, number_of, connection_array):
        weights = np.random.poisson(self.lambdas[0], number_of)
        return np.array([10 * weights[i - 1] if i == j else weights[i - 1] for i, j in connection_array])

    def sample_other(self, param_i, number_of):
        return np.random.poisson(self.parameters[param_i], number_of)

    def range(self, param_i):
        return 1/3 * abs(np.random.poisson(self.lambdas[param_i], 1) - np.random.poisson(self.lambdas[param_i], 1))

    def get_type(self):
        return "Poisson"


class Gaussian(Distribution):
    def __post_init__(self):
        self.means = self.parameters[0]
        self.std = self.parameters[1]

    def sample_weights(self, number_of, connection_array):
        weights = np.random.normal(self.means[0], self.std[0], number_of)
        return np.array([10 * weights[i - 1] if i == j else weights[i - 1] for i, j in connection_array])

    def sample_other(self, param_i, number_of):
        return np.random.normal(self.means[param_i], self.std[param_i], number_of)

    def range(self, param_i):
        return 1/3 * abs(np.random.normal(self.means[param_i], self.std[param_i], 1) - np.random.normal(self.means[param_i], self.std[param_i], 1))

    def get_type(self):
        return "Normal"

