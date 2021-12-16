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

        new_individual = Individual(new_genome, self.num_nodes)

        return new_individual

    def microbial_cross_over(self, weakest_individual, change_ratio=0.5):
        """ Performs microbial cross-over; some fraction of the weakest individual's genome is replaced by one of the
            stronger individual's genome"""

        length = len(weakest_individual.genome)
        num_swaps = np.floor(change_ratio * length)

        for _ in range(int(num_swaps)):
            pos = np.random.randint(0, length)
            weakest_individual.genome[pos] = self.genome[pos]

        return weakest_individual

    def fitness(self, target_signal, final_t, fitness_type):
        """ Hook method for the fitness of a individual"""

        self.ctrnn.node_values = np.array([0.0 for _ in range(self.num_nodes)])

        if fitness_type == "sample":
            return self.sample_fitness(target_signal, final_t)
        elif fitness_type == "simpsons":
            return self.simpsons_fitness(target_signal, final_t)

    def sample_fitness(self, target_signal, final_t):
        """ Evaluates the fitness level in a individual, true_signal is a function which
            self.ctrnn seeks to approximate. Done by sampling from the true signal and taking an error"""

        num_samples = 10
        step_size = final_t / num_samples
        fitness = 0

        for idx in range(num_samples):
            sample_t = step_size * idx
            fitness += abs(target_signal(sample_t) - self.evaluate(sample_t)[0])

        return (1/num_samples) * fitness

    def simpsons_fitness(self, target_signal, final_t):
        """ Returns fitness value by preform integral of true_signal - simpson's approximation of prediction values.
            This is a approximation of the area between the prediction and actual curve"""

        import scipy.integrate as sci

        self.ys = np.array([0.0 for _ in range(self.num_nodes)])
        self.last_time = 0
        step_size = 0.01
        num_samples = int(final_t / step_size)
        t = []
        y = []

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

        if self.last_time == final_t:
            return self.ys[-1]

        while self.last_time < final_t:

            for node in range(self.num_nodes):
                self.ctrnn.set_input(node, self.input_signal(self.last_time))

            self.ctrnn.update()
            self.last_time += self.ctrnn.step_size

        if not self.ctrnn.save:
            self.ctrnn.node_values = [0.0 for _ in range(self.num_nodes)]
            self.last_time = 0

        # assuming that the last node is the only output node
        return self.ctrnn.node_values[-1]

