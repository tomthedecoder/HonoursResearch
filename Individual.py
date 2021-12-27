from CTRNN import CTRNN
import numpy as np


def I(t):
    return np.sin(t)


class Individual:
    def __init__(self, genome, num_nodes, center_crossing, connection_array=None, num_weights=None):
        """ An individual of some environment"""

        self.genome = genome
        self.num_nodes = num_nodes
        self.ctrnn = CTRNN(num_nodes=self.num_nodes, genome=self.genome, connection_array=connection_array,
                           num_weights=num_weights, center_crossing=center_crossing)

        # input to each node of this individual
        self.input_signal = lambda t: I(t)

        # the linear rank of this individual, relative to fitness of other individuals. Between 0 and 2.
        self.rank_score = 0

        # last computed fitness value
        self.last_fitness = 0

        # determines if the fitness value still valid
        self.fitness_valid = False

        # times of evaluation
        self.last_time = 0

    def normal_cross_over(self, other_individual):
        """ Returns a new genotype from uniformly choosing genes from existing two"""

        num_swaps = int(np.ceil(len(self.genome) / 2))

        # deep copy
        new_genome = []
        for gene in self.genome:
            new_genome.append(gene)

        for idx in range(num_swaps):
            pos = np.random.randint(0, len(self.genome))
            new_genome[pos] = other_individual.genome[pos]

        # fitness valid is False by default
        new_individual = Individual(new_genome, self.num_nodes, False)

        return new_individual

    def microbial_cross_over(self, weakest_individual, change_ratio):
        """ Performs microbial cross-over; some fraction of the weakest individual's genome is replaced by one of the
            stronger individual's genome"""

        length = len(weakest_individual.genome)
        num_swaps = np.floor(3/2 * change_ratio * length + 1)

        for _ in range(int(num_swaps)):
            pos = np.random.randint(0, length)
            weakest_individual.change_parameter(pos, self.genome[pos])

        weakest_individual.fitness_valid = False
        return weakest_individual

    def cross_over(self, individual, type="microbial", *args):
        """ Hook method for cross-over"""

        if type == "microbial":
            return self.microbial_cross_over(individual, args[0])
        elif type == "normal":
            return self.normal_cross_over(individual)

    def fitness(self, target_signal, fitness_type, *args):
        """ Hook method for the fitness of a individual"""

        self.ctrnn.node_values = np.array([0.0 for _ in range(self.num_nodes)])
        final_t = args[0]

        if self.fitness_valid:
            return self.last_fitness
        if fitness_type == "sample":
            return self.sample_fitness(target_signal, final_t)
        elif fitness_type == "simpsons":
            return self.simpsons_fitness(target_signal, final_t)
        else:
            raise ValueError("Invalid fitness type")

    def sample_fitness(self, target_signal, final_t):
        """ Evaluates the fitness level in a individual, true_signal is a function which
            self.ctrnn seeks to approximate. Done by sampling from the true signal and taking an error"""

        num_samples = 10
        step_size = final_t / num_samples
        fitness = 0

        # reset CTRNN
        self.ctrnn.save = True
        E = self.evaluate(final_t=final_t)
        self.ctrnn.save = False

        for idx in range(num_samples):
            sample_t = step_size * idx
            fitness += abs(target_signal(sample_t) - E[idx])

        return (1/num_samples) * fitness

    def simpsons_fitness(self, target_signal, final_t):
        """ Returns fitness value by preform integral of true_signal - simpson's approximation of prediction values.
            This is a approximation of the area between the prediction and actual curve"""

        import scipy.integrate as sci

        step_size = 0.01
        num_samples = int(final_t / step_size)
        t = []
        y = []

        # reset CTRNN
        self.ctrnn.reset()
        self.last_time = 0
        self.ctrnn.save = True
        self.evaluate(final_t=final_t)
        E = self.ctrnn.node_history[-1]
        self.ctrnn.save = False

        for idx in range(num_samples):
            t.append(idx * step_size)
            y.append(abs(E[idx] - target_signal(t[-1])))

        return np.float(sci.simps(y=y, x=t))

    def set_rank(self, new_rank):
        self.rank_score = new_rank

    def get_rank(self):
        return self.rank_score

    def evaluate(self, final_t):
        """ Return evaluation starting at a single time step."""

        self.ctrnn.reset()
        self.last_time = 0

        if self.last_time > final_t:
            raise ValueError("Current time greater than final time")

        num_steps = abs(final_t - self.last_time) // self.ctrnn.step_size
        for _ in range(int(num_steps)):
            self.ctrnn.set_forcing(self.input_signal(self.last_time))
            self.ctrnn.update()
            self.last_time += self.ctrnn.step_size

        # assuming that the last node is the only output node
        return self.ctrnn.node_values[-1]

    def change_parameter(self, pos, new_value):
        """ Sets the position in the genome in the new value and also updates the ctrnn"""

        self.genome[pos] = new_value
        self.ctrnn.set_parameter(pos, new_value)

