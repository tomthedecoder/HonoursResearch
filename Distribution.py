import numpy as np
from dataclasses import dataclass, field


def uniform_parameters():
    """ Returns lows and highs arrays for distribution class based off maximum values of parameters"""

    tau_low = 0.5
    tau_high = 5
    bias_low = -16
    bias_high = 16
    diag_weight_low = -16
    diag_weight_high = 16
    off_weight_low = -16
    off_weight_high = 16
    input_weight_low = -0.8
    input_weight_high = 0.8

    return [[diag_weight_low, off_weight_low, tau_low, bias_low, input_weight_low],
            [diag_weight_high, off_weight_high, tau_high, bias_high, input_weight_high]]


def poisson_parameters():
    diag_weight_lambda = 10
    off_weight_lambda = 1
    tau_lambda = 1
    bias_lambda = 5
    input_weight_lambda = 5
    shift_lambda = 1

    return [diag_weight_lambda, off_weight_lambda, tau_lambda, bias_lambda, input_weight_lambda, shift_lambda]


def normal_parameters():
    diag_weight_mu = 5
    diag_weight_std = 1
    off_weight_mu = 1
    off_weight_std = 0.5
    tau_mu = 0.6
    tau_std = 0.2
    bias_mu = -2
    bias_std = 0.5
    input_weight_mu = 0.6
    input_weight_std = 1
    shift_mu = 0.5
    shift_std = 0.15

    return [[diag_weight_mu, off_weight_mu, tau_mu, bias_mu, input_weight_mu, shift_mu],
            [diag_weight_std, off_weight_std, tau_std, bias_std, input_weight_std, shift_std]]


def random_sgn():
    return [-1, 1][np.random.randint(0, 2)]


@dataclass(unsafe_hash=True)
class Distribution:
    """ Abstract/virtual type class, represents the probability of some parameter being initialized."""

    parameters: list = field(init=True)

    def __post_init__(self):
        self.num_parameters = len(self.parameters)

    def sample_weights(self, connection_array):
        weights = np.array([])
        for idx, conn in enumerate(connection_array):
            i, j = conn
            if i == j:
                weight = self.sample_other(1, 1)
            else:
                weight = self.sample_other(0, 1)

            weights = np.append(weights, weight)

        return weights

    def sample_other(self, param_i, number_of):
        return np.array([])

    def range(self, param_i):
        return abs(self.sample_other(param_i, 1) - self.sample_other(param_i, 1))

    @staticmethod
    def get_type(self):
        return "virtual distribution"

    @staticmethod
    def make_distribution(distribution_type):
        """ A factory method to create a distribution (non-virtual)"""

        distribution_type = distribution_type.strip().lower()

        if distribution_type == "uniform":
            parameters = uniform_parameters()
            return Uniform(parameters)
        elif distribution_type == "poisson":
            parameters = poisson_parameters()
            return Poisson(parameters)
        elif distribution_type == "normal":
            parameters = normal_parameters()
            return Normal(parameters)
        else:
            raise ValueError("Invalid distribution type")


class Uniform(Distribution):

    def __post_init__(self):
        self.lows = self.parameters[0]
        self.highs = self.parameters[1]

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
        return "uniform"


class Poisson(Distribution):

    def __post_init__(self):
        self.lambdas = self.parameters

    def sample_other(self, param_i, number_of):
        return random_sgn() * np.random.poisson(self.lambdas[param_i], number_of)

    def get_type(self):
        return "poisson"


class Normal(Distribution):
    def __post_init__(self):
        self.means = self.parameters[0]
        self.std = self.parameters[1]

    def sample_other(self, param_i, number_of):
        return random_sgn() * np.random.normal(self.means[param_i], self.std[param_i], number_of)

    def range(self, param_i):
        return random_sgn() * (self.means[param_i] + 3 * self.std[param_i])

    def get_type(self):
        return "normal"
