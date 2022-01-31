import numpy as np


class Individual(object):
    def __init__(self, genome, num_nodes, input_signals):
        """ An individual of some environment"""

        self.genome = genome
        self.num_genes = len(self.genome)
        self.num_nodes = num_nodes

        # the linear rank of this individual, relative to fitness of other individuals. Between 0 and 2.
        self.rank_score = 0

        # last computed fitness value
        self.last_fitness = 0

        # determines if the fitness value still valid
        self.fitness_valid = False

        self.input_signals = input_signals
        self.num_inputs = len(self.input_signals)

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
        new_individual = Individual(new_genome, self.num_nodes)

        return new_individual

    def microbial_cross_over(self, other_individual, change_ratio):
        """ Performs microbial cross-over; some fraction of the weakest individual's genome is replaced by one of the
            stronger individual's genome"""

        length = len(other_individual.genome)
        num_swaps = np.ceil(change_ratio * length)
        for _ in range(int(num_swaps)):
            pos = np.random.randint(0, length)
            for _ in range(100):
                if other_individual.genome[pos] != self.genome[pos]:
                    break
                pos = np.random.randint(0, length)

            other_individual.set_parameter(pos, self.genome[pos])

        other_individual.fitness_valid = False

        return other_individual

    def cross_over(self, individual, cross_over_type="microbial", *args):
        """ Hook method for cross-over"""

        if cross_over_type == "microbial":
            return self.microbial_cross_over(individual, args[0])
        elif cross_over_type == "normal":
            return self.normal_cross_over(individual)

    def fitness(self, target_signal, fitness_type, *args):
        """ Hook method for the fitness of an individual"""

        final_t = args[0]

        if self.fitness_valid:
            fitness = self.last_fitness
        elif fitness_type == "sample":
            fitness = self.sample_fitness(target_signal, final_t)
        elif fitness_type == "simpsons":
            fitness = self.simpsons_fitness(target_signal, final_t)
        else:
            raise ValueError("Invalid fitness type")

        self.fitness_valid = True
        self.last_fitness = fitness
        return fitness

    def sample_fitness(self, target_signal, final_t):
        """ Evaluates the fitness level in a individual, true_signal is a function which self.ctrnn seeks to approximate.
            Done by sampling from the true signal and taking an error"""

        num_samples = 10
        step_size = final_t / num_samples
        fitness = 0

        E = self.evaluate(final_t)

        for idx in range(num_samples):
            sample_t = step_size * idx
            fitness += abs(E[idx] - target_signal(sample_t))

        return (1/num_samples) * fitness

    def simpsons_fitness(self, target_signal, final_t):
        """ Returns fitness value by preform integral of target_signal - prediction, the simpson's approximation of
            prediction values. This is a approximation of the area between the prediction and actual curve"""

        import scipy.integrate as sci

        t = []
        y = []

        times, predictions = self.evaluate(final_t)
        for idx, value in enumerate(predictions):
            t.append(times[idx])
            y.append(abs(value - target_signal(times[idx])))

        return np.float(sci.simpson(y=y, x=t))

    def set_rank(self, new_rank):
        self.rank_score = new_rank

    def get_rank(self):
        return self.rank_score

    def evaluate(self, *args):
        """ Evaluate the individual, i.e assess it's behaviour subject to constraint(s) *args"""

        return []

    def set_parameter(self, pos, new_value):
        """ Sets the parameter at index pos to new_value"""

        pass
