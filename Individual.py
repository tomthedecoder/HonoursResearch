import numpy as np
from scipy.fft import fft, fftfreq
import scipy.integrate as sci
from TargetSignal import TargetSignal


class Individual:
    def __init__(self, individual_parameters):
        """ Member of an environment which can reproduce with other individuals. Also provides methods
            to compute fitness level"""

        self.params = individual_parameters
        self._rank = 0
        self.last_fitness = 0

    def cross_over(self, individual, cross_over_type="microbial", *args):
        """ Hook method for cross-over"""

        if cross_over_type == "microbial":
            return self.microbial_cross_over(individual, args[0])
        elif cross_over_type == "normal":
            return self.normal_cross_over(individual)
        else:
            raise ValueError("Unrecognised cross over type")

    def normal_cross_over(self, other_individual):
        """ Returns a new genotype by uniformly choosing genes from existing two"""

        num_swaps = int(np.ceil(len(self.params.genome) / 2))

        for idx in range(num_swaps):
            pos = np.random.randint(0, self.params.num_genes)
            other_individual.\
                set_parameter(pos, self.params.genome[pos])

        return other_individual

    def microbial_cross_over(self, other_individual, change_ratio):
        """ Performs microbial cross-over; some fraction of the weakest individual's genome is replaced by one of the
            stronger individual's genome"""

        length = len(other_individual.params.genome)
        num_swaps = np.ceil(change_ratio * length)
        for _ in range(int(num_swaps)):
            pos = np.random.randint(0, length - 1)
            for _ in range(100):
                if other_individual.params.genome[pos] != self.params.genome[pos]:
                    break
                pos = np.random.randint(0, length)

            other_individual.params.set_parameter(pos, self.params.genome[pos])

        return other_individual

    def fitness(self, target_signal: TargetSignal, times: np.array, y_output: np.array, fitness_type: str):
        """ Hook method for the fitness of an individual"""

        if fitness_type == "sample":
            fitness = self.sample_fitness(target_signal, times, y_output)
        elif fitness_type == "simpsons":
            fitness = self.simpsons_fitness(target_signal, times, y_output)
        elif fitness_type == "inflection":
            fitness = self.inflection_fitness(target_signal, times, y_output)
        elif fitness_type == "1/simpsons":
            fitness = self.inv_simpsons_fitness(target_signal, times, y_output)
        else:
            raise ValueError("Invalid fitness type")

        self.last_fitness = fitness

        return fitness

    def inv_simpsons_fitness(self, target_signal: TargetSignal, times: np.array, y_output: np.array):
        return -1/self.simpsons_fitness(target_signal, times, y_output)

    def inflection_fitness(self, target_signal: TargetSignal, times: np.array, y_output: np.array):
        """ Returns the absolute value of the number of times the gradient of the ctrnn output curve changes sign
            - the number of times the target signal curve does"""

        def center_derivative(values, n):
            return values[n+1] - values[n-1]

        def forward_derivative(values, n):
            return values[n+1] - values[n]

        def backward_derivative(values, n):
            return values[n] - values[n-1]

        def number_of_inflections(values):
            """ Counts the number of times the gradient, which is numerically approximated, changes sign"""

            length = len(values)

            def sgn(y, ls):
                m = 0.01
                if y >= m:
                    return 1
                elif -m < y < m:
                    return ls
                else:
                    return -1

            step = 5
            last_sign = sgn(forward_derivative(values, 0), 0)
            num_inflections = 0
            for n in range(0, length, step):
                if n < length - 1:
                    new_sign = sgn(center_derivative(values, n), last_sign)
                else:
                    new_sign = sgn(backward_derivative(values, n), last_sign)

                if new_sign != last_sign:
                    num_inflections += 1

                last_sign = new_sign

            return num_inflections

        y_target = target_signal(times)

        return -abs(number_of_inflections(y_target) - number_of_inflections(y_output))

    def sample_fitness(self, target_signal: TargetSignal, times: np.array, y_output: np.array):
        """ Evaluates the fitness level of an individual by sampling from the true signal and taking an error"""

        num_points = len(times)
        y_target = target_signal(times)
        fitness = 0
        for i in range(num_points):
            fitness += abs(y_target[i] - y_output[i])

        return (-1/num_points) * fitness

    def simpsons_fitness(self, target_signal: TargetSignal, times: np.array, y_output: np.array):
        """ Returns fitness value by approximating the integral of target_signal - output| with the simpson's scheme
            This is an approximation of the area between the output and target curve"""

        y_abs = []
        y_target = target_signal(times)
        for idx, value in enumerate(y_output):
            y_abs.append(abs(value - y_target[idx]))

        return -sci.simpson(y=y_abs, x=times)

    @property
    def rank(self):
        return self._rank

    @rank.setter
    def rank(self, new_rank):
        self._rank = new_rank


