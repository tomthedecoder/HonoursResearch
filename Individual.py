import numpy as np
from scipy.fft import fft, fftfreq


class Individual(object):
    def __init__(self, individual_parameters):
        """ Member of an environment which can reproduce with other individuals. Also provides methods
            to compute fitness level"""

        self.params = individual_parameters
        self._rank = 0
        self.last_fitness = 0
        self.fitness_valid = False

    def normal_cross_over(self, other_individual):
        """ Returns a new genotype by uniformly choosing genes from existing two"""

        num_swaps = int(np.ceil(len(self.params.genome) / 2))

        for idx in range(num_swaps):
            pos = np.random.randint(0, self.params.num_genes)
            other_individual.set_parameter(pos, self.params.genome[pos])

        return other_individual

    def microbial_cross_over(self, other_individual, change_ratio):
        """ Performs microbial cross-over; some fraction of the weakest individual's genome is replaced by one of the
            stronger individual's genome"""

        length = len(other_individual.params.genome)
        num_swaps = np.ceil(change_ratio * length)
        for _ in range(int(num_swaps)):
            pos = np.random.randint(0, length)
            for _ in range(100):
                if other_individual.params.genome[pos] != self.params.genome[pos]:
                    break
                pos = np.random.randint(0, length)

            other_individual.params.set_parameter(pos, self.params.genome[pos])

        other_individual.fitness_valid = False

        return other_individual

    def cross_over(self, individual, cross_over_type="microbial", *args):
        """ Hook method for cross-over"""

        if cross_over_type == "microbial":
            return self.microbial_cross_over(individual, args[0])
        elif cross_over_type == "normal":
            return self.normal_cross_over(individual)
        else:
            raise ValueError("Unrecognised cross over type")

    def fitness(self, target_signal, fitness_type, final_t):
        """ Hook method for the fitness of an individual"""

        if self.fitness_valid:
            fitness = self.last_fitness
        elif fitness_type == "weight":
            fitness = self.weight_fitness(target_signal, final_t)
        elif fitness_type == "forcing":
            fitness = self.forcing_weight_fitness(target_signal, final_t)
        elif fitness_type == "sample":
            fitness = self.sample_fitness(target_signal, final_t)
        elif fitness_type == "simpsons":
            fitness = self.simpsons_fitness(target_signal, final_t)
        elif fitness_type == "fourier":
            fitness = self.fourier_fitness(target_signal, final_t)
        elif fitness_type == "inflection":
            fitness = self.inflection_fitness(target_signal, final_t)
        elif fitness_type == "inflection&simpsons":
            fitness = (self.inflection_fitness(target_signal, final_t) + 1) * (self.simpsons_fitness(target_signal, final_t))
        elif fitness_type == "fourier&simpsons":
            fitness = self.simpsons_fitness(target_signal, final_t) / (self.fourier_fitness(target_signal, final_t) + 1)
        elif fitness_type == "inflection&fourier&simpsons":
            coefficient = (self.inflection_fitness(target_signal, final_t) + 1) / (self.fourier_fitness(target_signal, final_t) + 1)
            fitness = coefficient * self.simpsons_fitness(target_signal, final_t)
        else:
            raise ValueError("Invalid fitness type")

        self.fitness_valid = True
        self.last_fitness = fitness

        return fitness

    def weight_fitness(self, target_signal, final_t):
        total = 0
        for weight in self.params.weights:
            total += abs(weight.value)

        return -total

    def forcing_weight_fitness(self, target_signal, final_t):
        total = 0
        for i, tau in enumerate(self.params.taus):
            bias = self.params.biases[i]
            total += bias / tau

        return total

    def inflection_fitness(self, target_signal, final_t):
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

        times, y_output = self.evaluate(final_t)
        y_target = target_signal(times)

        return abs(number_of_inflections(y_target) - number_of_inflections(y_output))

    def fourier_fitness(self, target_signal, final_t):
        """ Takes the difference between where the peek of each fourier transform is located.
            Effectively difference in frequency between waves. Assumes both waves have only one peek."""

        def find_max(values):
            max_pos = 0
            maximum = values[0]
            for pos, val in enumerate(values):
                if val > maximum:
                    max_pos = pos
                    maximum = val

            return max_pos, maximum

        return 0

    def sample_fitness(self, target_signal, final_t):
        """ Evaluates the fitness level of an individual by sampling from the true signal and taking an error"""

        fitness = 0
        times, y_output = self.evaluate(final_t)
        y_target = target_signal(times)
        num_samples = len(times)
        for i in range(num_samples):
            fitness += abs(y_target[i] - y_output[i])

        return (1/num_samples) * fitness

    def simpsons_fitness(self, target_signal, final_t):
        """ Returns fitness value by approximating the integral of |target_signal - output| with the simpson's scheme
            This is an approximation of the area between the output and target curve"""

        import scipy.integrate as sci

        t = []
        y = []
        times, predictions = self.evaluate(final_t)
        # ignore times before this, so that flow can reach it's limit cycle
        start_time = 10 * times[-1] / 100
        for idx, value in enumerate(predictions):
            if times[idx] < start_time:
                pass
            t.append(times[idx])
            y.append(abs(value - target_signal(times[idx])))

        return np.float(sci.simpson(y=y, x=t))

    @property
    def rank(self):
        return self._rank

    @rank.setter
    def rank(self, new_rank):
        self._rank = new_rank

    def evaluate(self, *args):
        """ Evaluate the individual, i.e assess it's behaviour subject to constraint(s) *args"""

        return []

