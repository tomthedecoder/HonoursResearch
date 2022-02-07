import numpy as np
from CTRNN import CTRNN
from CTRNNStructure import *
from CTRNNParameters import *
from Distribution import *
import inspect


class Environment:
    def __init__(self, target_signal, ctrnn_structure, pop_size=3, mutation_chance=0.9):
        """ A container, which holds individuals of the environment, methods for evolution and parameters of the
            experiment initialises individuals along some Gaussian distribution or takes a pre-existing array of
            individuals"""

        self.pop_size = pop_size
        self.mutation_chance = mutation_chance
        self.struct = ctrnn_structure
        self.mutation_coefficient = 0.1
        self.individuals = []
        self.target_signal = target_signal

    def fill_individuals(self, output_handler=None, forcing_signals=None, individuals=None):
        """ Initialise self.individuals using uniform distributions. If a individual array is provided, simply assign
            this to self.individuals"""

        if individuals is None and (output_handler is None or forcing_signals is None):
            raise ValueError("When the individuals is the default value, an output handler "
                             "and forcing signal argument must be passed in")

        if individuals is None:
            # populate self.individuals with CTRNNs
            self.individuals = []
            for _ in range(self.pop_size):
                genome = self.make_genome(len(forcing_signals[0]))
                ctrnn_parameters = CTRNNParameters(genome, output_handler, forcing_signals, self.struct.connection_array)
                ctrnn = CTRNN(ctrnn_parameters)
                self.individuals.append(ctrnn)
        else:
            self.individuals = individuals

    def make_genome(self, num_forcing):
        """ Generates a new genome from the environment's specifications"""

        weights = np.array(self.struct.distribution.sample_weights(self.struct.connection_array))
        taus = np.array(self.struct.distribution.sample_other(2, self.struct.num_nodes))
        forcing_weights = np.array(self.struct.distribution.sample_other(4, self.struct.num_nodes * num_forcing))

        if self.struct.center_crossing:
            biases = np.array([0.0 for _ in range(self.struct.num_nodes)])
            for wi, weight in enumerate(weights):
                i, j = self.struct.connection_array[wi]
                i -= 1
                j -= 1
                biases[j] += weight
            biases = np.divide(biases, -2)
        else:
            biases = np.array(self.struct.distribution.sample_other(3, self.struct.num_nodes))

        genome = np.append(weights, taus)
        genome = np.append(genome, biases)
        genome = np.append(genome, forcing_weights)

        return genome

    def mutate(self):
        """ Preforms mutation on a random part of the genome of a random individual by drawing from the
            respective distribution"""

        index = np.random.randint(0, self.pop_size-1)
        individual = self.individuals[index]

        genome_length = individual.params.num_genes
        pos = np.random.randint(0, genome_length)
        direction = [-1, 1][np.random.randint(0, 2)]

        # assign random value from proper distribution
        mutation_distance = direction * self.mutation_coefficient
        if pos < individual.params.num_nodes:
            mutation_distance *= self.struct.distribution.range(0)
        elif pos < individual.params.num_weights:
            mutation_distance *= self.struct.distribution.range(1)
        elif pos < individual.params.num_weights + individual.params.num_nodes:
            mutation_distance *= self.struct.distribution.range(2)
            if mutation_distance == 0:
                mutation_distance = 0.01
        elif pos < individual.params.num_weights + 2 * individual.params.num_nodes:
            mutation_distance *= self.struct.distribution.range(3)
        elif pos < individual.params.num_weights + individual.params.num_nodes * (2 + individual.params.num_forcing):
            mutation_distance *= self.struct.distribution.range(4)

        individual.params.set_parameter(pos, individual.params.genome[pos] + mutation_distance)

    def rank(self, final_t, fitness_type):
        """ Assign rank between 0 and 2 to each individual in the environment. The fitter an individual the higher
            it's rank"""

        step_size = 1 / self.pop_size

        # assess fitness levels
        for idx, individual in enumerate(self.individuals):
            self.individuals[idx].last_fitness = individual.fitness(self.target_signal, fitness_type, final_t)

        # sort individuals
        self.individuals = sorted(self.individuals, key=lambda i: i.last_fitness, reverse=True)

        # assign rank
        for idx, individual in enumerate(self.individuals):
            new_rank = 2 * (idx + 1) * step_size
            individual.rank = new_rank

    def lower_third_reproduction(self, final_t, cross_over_type="microbial", fitness_type="simpsons"):
        """ Performs a round of reproduction. The lower third is replaced with off-spring from the top third"""

        self.rank(final_t, fitness_type)

        # get lower and upper third
        l = int(np.floor(1 / 3 * len(self.individuals)))
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
            self.individuals[0] = best_individual.microbial_cross_over(weakest_individual, change_ratio=0.6)

    def save_state(self, environment_index, state_file="state_file"):
        """ Writes genome and connection matrix of CTRNN to file. Format is
        # number of individuals
        # connection matrix
        # the true signal and then
        # distribution parameters
        # _
        # fitness_valid, last_fitness, genome
        # _
        # for every individual"""

        state_file = f"{state_file}{environment_index}"

        some_ctrnn = self.individuals[0]

        # collect string to store in file
        num_nodes = some_ctrnn.params.num_nodes
        contents = "{}\n{}\n".format(self.pop_size, num_nodes)

        # store self.true_signal
        signal_as_string = inspect.getsourcelines(self.target_signal)[0][0]
        contents += signal_as_string[signal_as_string.find(" ", 2):].strip() + '\n'

        # add the connection matrix to write string
        weights = some_ctrnn.params.weights

        for weight in weights:
            i = weight.traveled_from()
            j = weight.traveled_to()
            contents += "{},{} ".format(i, j)
        contents += '\n'

        # output handler method
        contents += f"{some_ctrnn.params.output_handler.method.strip()}\n"

        # center crossing
        contents += f"{self.struct.center_crossing}\n"

        # connection type
        contents += f"{self.struct.connection_type.strip()}\n"

        # distribution parameters
        contents += f"{self.struct.distribution.get_type().strip()} {str(self.struct.distribution).strip()}\n"

        # add fitness_valid, fitness, genome to write string
        for idx, individual in enumerate(self.individuals):
            str_individual = f"{individual.fitness_valid} {individual.last_fitness} "
            for idx, gene in enumerate(individual.params.genome):
                str_individual += str(gene) + " "
            contents += str_individual + "\n"

        # write string to file
        with open(state_file, "w") as write_file:
            write_file.write(contents)

    @staticmethod
    def load_environment(environment_index, forcing_signals, state_file="state_file"):
        """ Returns an environment filled with CTRNNs from a saved state"""

        state_file = f"{state_file}{environment_index}"

        # read file contents
        with open(state_file) as read_file:
            contents = read_file.readlines()

        pop_size = int(contents[0])
        num_nodes = int(contents[1])
        target_signal = lambda t: t ** 100

        # line contains connection array
        line = contents[3].split()
        connection_array = []
        for item in line:
            p = int(item.find(","))
            i = int(item[0:p])
            j = int(item[p + 1:].strip())
            connection_array.append((i, j))

        handler_method = contents[4]
        output_handler = OutputHandler(handler_method)

        center_crossing = bool(contents[5])
        connection_type = contents[6]
        start = 7

        # contains the parameters for the distribution
        r = contents[start].find(" ")
        distribution_type = contents[start][0:r].strip().lower()
        parameters = [float(y) for x in contents[start][r:].split() for y in x.split(',')]
        if distribution_type == "uniform":
            lows = [x for i, x in enumerate(parameters) if i % 2 == 0]
            highs = [x for i, x in enumerate(parameters) if i % 2 == 1]
            distribution = Uniform([lows, highs])
        elif distribution_type == "poisson":
            distribution = Poisson(parameters)
        elif distribution_type == "gaussian":
            distribution = Normal(parameters)
        else:
            raise ValueError("Invalid distribution type in save file")

        # get genomes
        individuals = []
        for line in contents[start+1:]:
            genome = []

            # add fitness valid, fitness true
            p1 = line.find(" ")
            p2 = line.find(" ", p1 + 1)
            fitness_valid = bool(line[0:p1])
            last_fitness = float(line[p1:p2])

            num = ""
            for idx, char in enumerate(line):
                if idx <= p2:
                    continue
                if char == " ":
                    genome.append(float(num))
                    num = ""
                    continue
                num += char

            ctrnn_parameters = CTRNNParameters(genome, output_handler, forcing_signals, connection_array)
            ctrnn = CTRNN(ctrnn_parameters)
            ctrnn.fitness_valid = fitness_valid
            ctrnn.last_fitness = last_fitness
            individuals.append(ctrnn)

        ctrnn_structure = CTRNNStructure(distribution, num_nodes, connection_array, connection_type, center_crossing)
        new_environment = Environment(target_signal, ctrnn_structure, pop_size)
        new_environment.fill_individuals(individuals=individuals)

        return new_environment
