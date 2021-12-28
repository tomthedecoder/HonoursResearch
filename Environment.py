import numpy as np
from CTRNN import CTRNN
import inspect


class Environment:
    def __init__(self, target_signal, pop_size=1000, weight_high=5, weight_low=-5, tau_low=0, tau_high=5, bias_low=-2, bias_high=2, center_crossing=False, mutation_chance=0.05):
        """ A container, which holds individuals of the environment, methods for evolution and parameters of the
            experiment initialises individuals along some Gaussian distribution or takes a pre-existing array of
            individuals"""

        self.pop_size = pop_size
        self.mutation_chance = mutation_chance

        # ranges for the uniform distribution
        self.weight_high = weight_high
        self.weight_low = weight_low
        self.tau_low = tau_low
        self.tau_high = tau_high
        self.bias_high = bias_high
        self.bias_low = bias_low
        self.center_crossing = center_crossing

        self.individuals = []

        self.target_signal = target_signal

    def fill_individuals(self, num_nodes, connection_array=None, individuals=None):
        """ Initialise self.individuals using uniform distributions. If a individual array is provided, simply assign
            this to self.individuals"""

        if individuals is None:
            # if no connection array is passed in, put a connection between all nodes
            if connection_array is None:
                connection_array = []
                for idx in range(num_nodes):
                    for idy in range(num_nodes):
                        connection_array.append((idx + 1, idy + 1))

            # populate self.individuals with CTRNNs
            self.individuals = []
            for _ in range(self.pop_size):
                genome = self.make_genome(num_nodes, connection_array)
                self.individuals.append(CTRNN(num_nodes, genome, connection_array))
        else:
            self.individuals = individuals

    def make_genome(self, num_nodes, connection_array):
        """ Generates a new genome from the environment's specifications"""

        num_weights = len(connection_array)

        weights = np.random.uniform(self.weight_low, self.weight_high, size=num_weights)
        taus = np.random.uniform(self.tau_low+0.01, self.tau_high, size=num_nodes)

        # create center crossing biases here if called for
        if self.center_crossing:
            # this procedure is possible because the weight in weights[idx]
            # corresponds to the edge in connection_array[idx]
            biases = np.array([0.0 for _ in range(num_nodes)])
            for idx in range(num_weights):
                i, j = connection_array[idx]
                i -= 1
                biases[i] += weights[idx] / 2
        else:
            biases = np.random.uniform(self.bias_low, self.bias_high, size=num_nodes)

        genome = np.append(weights, taus)
        genome = np.append(genome, biases)

        return genome

    def mutate(self):
        """ Preforms mutation on a random part of the genome of a random individual by drawing from the
            respective distribution"""

        index = np.random.randint(0, self.pop_size)
        individual = self.individuals[index]

        genome_length = len(individual.genome)
        pos = np.random.randint(0, genome_length)

        # assign random value from proper distribution
        if pos <= individual.num_nodes ** 2 - 1:
            individual.set_parameter(pos, np.random.randint(self.weight_low, self.weight_high))
        elif individual.num_nodes ** 2 - 1 < pos <= individual.num_nodes ** 2 + individual.num_nodes - 1:
            individual.set_parameter(pos, np.random.randint(self.tau_low, self.tau_high))
            if individual.genome[pos] == 0:
                individual.set_parameter(pos, 0.001)
        else:
            individual.set_parameter(pos, np.random.randint(self.bias_low, self.bias_high))

    def rank(self, final_t, fitness_type):
        """ Assign rank between 0 and 2 to each individual in the environment. The fitter an individual the higher
            it's rank"""

        step_size = 1/self.pop_size

        # assess fitness levels
        for idx, individual in enumerate(self.individuals):
            self.individuals[idx].last_fitness = individual.fitness(self.target_signal, fitness_type, final_t)
            self.individuals[idx].fitness_valid = True

        # sort individuals
        self.individuals = sorted(self.individuals, key=lambda i: i.last_fitness, reverse=True)

        # assign rank
        for idx, individual in enumerate(self.individuals):
            new_rank = 2 * (idx + 1) * step_size
            individual.set_rank(new_rank)

    def lower_third_reproduction(self, final_t, cross_over_type="normal", fitness_type="simpsons"):
        """ Performs a round of reproduction. The lower third is replaced with off-spring from the top third"""

        self.rank(final_t, fitness_type)

        # get lower and upper third
        l = int(np.floor(1/3 * len(self.individuals)))
        lower_third = self.individuals[0:l]
        length = len(self.individuals)

        # reassign parameters to individuals, lower individuals are at the start of self.individuals
        for idx, low_individual in enumerate(lower_third):

            p1id = np.random.randint(l, length)
            p2id = np.random.randint(l, length)

            parent1 = self.individuals[p1id]
            parent2 = self.individuals[p2id]

            if cross_over_type == "normal":
                new_individual = parent1.normal_cross_over(parent2)
            elif cross_over_type == "microbial":
                new_individual = parent1.microbial_cross_over(self.individuals[idx])
            else:
                raise ValueError("Specified fitness type is unknown")

            self.individuals[idx] = new_individual

    def weakest_individual_reproduction(self, final_t, cross_over_type="normal", fitness_type="simpsons"):
        """ Replaces the weakest individual with child of best two"""

        self.rank(final_t, fitness_type)

        best_individual = self.individuals[-1]
        second_individual = self.individuals[-2]
        weakest_individual = self.individuals[0]

        if cross_over_type == "normal":
            self.individuals[0] = best_individual.normal_cross_over(second_individual)
        elif cross_over_type == "microbial":
            self.individuals[0] = best_individual.microbial_cross_over(weakest_individual, change_ratio=0.5)

    def save_state(self, state_file="state_file"):
        """ Writes genome and connection matrix of CTRNN to file. Format is
        # number of individuals
        # connection matrix
        # the true signal and then
        # _
        # genome
        # _
        # for every individual"""

        # collect string to store in file
        num_nodes = self.individuals[0].num_nodes
        contents = "{}\n{}\n".format(self.pop_size, num_nodes)

        # store self.true_signal
        signal_as_string = inspect.getsourcelines(self.target_signal)[0][0]
        contents += signal_as_string[signal_as_string.find(" ", 2):].strip() + '\n'

        # add the connection matrix to write string
        weights = self.individuals[0].weights

        for weight in weights:
            i = weight.get_i()
            j = weight.get_j()
            contents += "{},{} ".format(i, j)
        contents += '\n'

        # add genomes to write string
        for idx, individual in enumerate(self.individuals):
            str_genome = ""
            for idx, gene in enumerate(individual.genome):
                str_genome += str(gene) + " "
            contents += str_genome + "\n"

        # write string to file
        with open(state_file, "w") as write_file:
            write_file.write(contents)

    @staticmethod
    def load_environment(filename="state_file"):
        """ Returns an environment filled with CTRNNs from a saved state"""

        # read file contents
        with open(filename) as read_file:
            contents = read_file.readlines()

        pop_size = int(contents[0])
        num_nodes = int(contents[1])
        true_signal = eval(contents[2][contents[2].find(" ", 2)+2:].strip())

        # line contains connection array
        line = contents[3].split()
        connection_array = []
        for item in line:
            p = int(item.find(","))
            i = int(item[0:p])
            j = int(item[p+1:].strip())
            connection_array.append((i, j))

        # get genomes
        individuals = []
        for line in contents[3+num_nodes:]:
            genome = []
            num = ""
            for char in line:
                if char == " ":
                    genome.append(float(num))
                    num = ""
                    continue
                num += char
            individuals.append(CTRNN(genome, num_nodes, connection_array))

        new_environment = Environment(true_signal, pop_size)
        new_environment.fill_individuals(num_nodes, connection_array, individuals)

        return new_environment









