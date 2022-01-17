import numpy as np
from CTRNN import CTRNN
from Distribution import Distribution
import inspect


class Environment:
    def __init__(self, target_signal, distribution, pop_size=3, center_crossing=False, mutation_chance=0.9):
        """ A container, which holds individuals of the environment, methods for evolution and parameters of the
            experiment initialises individuals along some Gaussian distribution or takes a pre-existing array of
            individuals"""

        self.pop_size = pop_size
        self.mutation_chance = mutation_chance
        self.distribution = distribution
        self.center_crossing = center_crossing
        self.mutation_coefficient = 0.05
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
                ctrnn = CTRNN(num_nodes, genome, connection_array)

                # sets biases to center-crossing
                if self.center_crossing:
                    biases = np.array([0.0 for _ in range(num_nodes)])
                    ctrnn.set_forcing(0)
                    for weight_i in range(ctrnn.num_weights):
                        i, j = connection_array[weight_i]
                        biases[i - 1] = ctrnn.get_forcing(weight_i)
                        for weight in ctrnn.weights:
                            if weight.traveled_to() == i:
                                biases[i - 1] += weight.value
                        biases[i - 1] /= 2

                self.individuals.append(ctrnn)
        else:
            self.individuals = individuals

    def make_genome(self, num_nodes, connection_array):
        """ Generates a new genome from the environment's specifications"""

        num_weights = len(connection_array)

        weights = self.distribution.uniform(0, num_weights)
        taus = self.distribution.uniform(1, num_nodes)
        biases = self.distribution.uniform(2, num_nodes)
        input_weights = self.distribution.uniform(3, num_nodes)

        genome = np.append(weights, taus)
        genome = np.append(genome, biases)
        genome = np.append(genome, input_weights)

        return genome

    def mutate(self):
        """ Preforms mutation on a random part of the genome of a random individual by drawing from the
            respective distribution"""

        index = np.random.randint(0, self.pop_size-1)
        individual = self.individuals[index]

        genome_length = len(individual.genome)
        pos = np.random.randint(0, genome_length)
        mutation_distance = 0
        direction = [-1, 1][np.random.randint(0, 2)]

        # assign random value from proper distribution
        if pos <= individual.num_nodes ** 2 - 1:
            mutation_distance = direction * self.mutation_coefficient * self.distribution.range(0)
        elif pos <= individual.num_nodes ** 2 + individual.num_nodes - 1:
            mutation_distance = direction * self.mutation_coefficient * self.distribution.range(1)
            if mutation_distance == 0:
                mutation_distance = 0.001
        elif pos <= individual.num_nodes ** 2 + 2*individual.num_nodes - 1:
            mutation_distance = direction * self.mutation_coefficient * self.distribution.range(2)
        else:
            mutation_distance = direction * self.mutation_coefficient * self.distribution.range(3)

        individual.set_parameter(pos, individual.genome[pos] + mutation_distance)

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
            individual.set_rank(new_rank)

    def lower_third_reproduction(self, final_t, cross_over_type="normal", fitness_type="simpsons"):
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
            self.individuals[0] = best_individual.microbial_cross_over(weakest_individual, change_ratio=0.5)

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

        # collect string to store in file
        num_nodes = self.individuals[0].num_nodes
        contents = "{}\n{}\n".format(self.pop_size, num_nodes)

        # store self.true_signal
        signal_as_string = inspect.getsourcelines(self.target_signal)[0][0]
        contents += signal_as_string[signal_as_string.find(" ", 2):].strip() + '\n'

        # add the connection matrix to write string
        weights = self.individuals[0].weights

        for weight in weights:
            i = weight.traveled_from()
            j = weight.traveled_to()
            contents += "{},{} ".format(i, j)
        contents += '\n'

        # distribution parameters
        contents += f"{str(self.distribution)}\n"

        # add fitness_valid, fitness, genome to write string
        for idx, individual in enumerate(self.individuals):
            str_individual = f"{individual.fitness_valid} {individual.last_fitness} "
            for idx, gene in enumerate(individual.genome):
                str_individual += str(gene) + " "
            contents += str_individual + "\n"

        # write string to file
        with open(state_file, "w") as write_file:
            write_file.write(contents)

    @staticmethod
    def load_environment(environment_index, state_file="state_file"):
        """ Returns an environment filled with CTRNNs from a saved state"""

        state_file = f"{state_file}{environment_index}"

        # read file contents
        with open(state_file) as read_file:
            contents = read_file.readlines()

        pop_size = int(contents[0])
        num_nodes = int(contents[1])
#       target_signal = eval(contents[2][contents[2].find(" ", 2) + 2:].strip())
        target_signal = lambda t: np.sin(t)

        # line contains connection array
        line = contents[3].split()
        connection_array = []
        for item in line:
            p = int(item.find(","))
            i = int(item[0:p])
            j = int(item[p + 1:].strip())
            connection_array.append((i, j))

        # contains the parameters for the distribution
        parameters = [float(x) for x in contents[4].split()]
        lows = [x for i, x in enumerate(parameters) if i % 2 == 0]
        highs = [x for i, x in enumerate(parameters) if i % 2 == 1]
        distribution = Distribution(lows, highs)

        # get genomes
        individuals = []
        for line in contents[5:]:
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

            ctrnn = CTRNN(num_nodes, genome, connection_array)
            ctrnn.fitness_valid = fitness_valid
            ctrnn.last_fitness = last_fitness
            individuals.append(ctrnn)

        new_environment = Environment(target_signal, distribution, pop_size)
        new_environment.fill_individuals(num_nodes, connection_array, individuals)

        return new_environment
