import numpy as np
from CTRNN import CTRNN
from KuramotoStructure import *
from KuramotoParameters import *
from KuramotoOscillator import *
from CTRNNStructure import *
from CTRNNParameters import *
from Distribution import *
from TargetSignal import *
import inspect


class Deme:
    def __init__(self, target_signal: TargetSignal, structure, network_type: str, pop_size=3):
        """ A container, which holds individuals. Uses methods for evolution and holds parameters of the
            experiment. Initialises individuals with specified distribution or takes existing array of individuals"""

        self.pop_size = pop_size
        self.struct = structure
        self.network_type = network_type
        self.individuals = []
        self.target_signal = target_signal
        self.mutation_std = 0.2
        self.mutation_mean = 0

    def fill_individuals(self, output_handler=None, forcing_signals=None, individuals=None):
        """ Initialise self.individuals using uniform distributions. If an individual array is provided, simply assign
            this to self.individuals"""

        if individuals is None and (output_handler is None or forcing_signals is None):
            raise ValueError("When the individuals is the default value, an output handler "
                             "and forcing signal arguments must be passed in")

        if individuals is None:
            # populate self.individuals with CTRNNs
            self.individuals = []
            for _ in range(self.pop_size):
                num_forcing = len(forcing_signals[0])
                if self.network_type == "ctrnn":
                    ctrnn_parameters = CTRNNParameters(self.make_ctrnn_genome(num_forcing), output_handler, forcing_signals, self.struct.connection_array)
                    individual = CTRNN(ctrnn_parameters)
                elif self.network_type == "kuramoto":
                    kuramoto_parameters = KuramotoParameters(self.make_kuramoto_genome(num_forcing), output_handler, forcing_signals, len(self.struct.connection_array))
                    individual = KuramotoOscillator(kuramoto_parameters, self.struct.connection_array)
                else:
                    raise ValueError("Invalid network type")

                self.individuals.append(individual)
        else:
            self.individuals = individuals

    def make_kuramoto_genome(self, num_forcing=1):
        """ Generates a genome for a kuramoto oscillator"""

        natural_frequencies = np.array(self.struct.distribution.sample_other(0, self.struct.num_nodes))
        k = np.array(self.struct.distribution.sample_other(1, self.struct.num_nodes * self.struct.num_nodes))

        genome = np.append(natural_frequencies, k)

        return genome

    def make_ctrnn_genome(self, num_forcing):
        """ Generates a new genome for a CTRNN network"""

        weights = np.array(self.struct.distribution.sample_weights(self.struct.connection_array))
        taus = np.array(self.struct.distribution.sample_other(2, self.struct.num_nodes))
        forcing_weights = np.array(self.struct.distribution.sample_other(4, self.struct.num_nodes * num_forcing))

        if self.struct.center_crossing:
            biases = np.array([0.0 for _ in range(self.struct.num_nodes)])
            for wi, weight in enumerate(weights):
                i, j = self.struct.connection_array[wi]
                biases[j] += weight
            biases = np.divide(biases, -2)
        else:
            biases = np.array(self.struct.distribution.sample_other(3, self.struct.num_nodes))

        genome = np.append(weights, taus)
        genome = np.append(genome, biases)
        genome = np.append(genome, forcing_weights)

        return genome

    def mutate(self):
        index = np.random.randint(0, self.pop_size-1)
        individual = self.individuals[index]

        genome_length = individual.params.num_genes
        pos = np.random.randint(0, genome_length)
        direction = [-1, 1][np.random.randint(0, 2)]

        mutation_distance = direction * np.random.normal(self.mutation_mean, self.mutation_std)
        individual.params.set_parameter(pos, individual.params.genome[pos] + mutation_distance)

        return index

    def rank(self, final_t, fitness_type):
        """ Assign rank between 0 and 2 to each individual in the environment and sort the individual array, with the
            highest ranked individuals at the end of the array."""

        # assess fitness levels, assign true fitness
        for idx, individual in enumerate(self.individuals):
            if not individual.params.eval_valid:
                times, y_output = individual.evaluate(final_t)

                self.individuals[idx].last_fitness = individual.fitness(self.target_signal, times, y_output, fitness_type)

        # sort individuals
        self.individuals = sorted(self.individuals, key=lambda i: i.last_fitness, reverse=False)

        # assign rank
        for idx, individual in enumerate(self.individuals):
            individual.rank = idx + 1

    def sink(self, individual_i: int, final_t: float, fitness_type: str):
        """ O(n) implementation to place individual_i in the correct location"""

        # determines if position i ordered

        def in_order(i, array):
            if i == 0:
                return array[0].last_fitness < array[1].last_fitness
            elif i == len(array) - 1:
                return array[-2].last_fitness < array[-1].last_fitness
            else:
                return array[i - 1].last_fitness < array[i].last_fitness < array[i + 1].last_fitness

        # make sure the individual to sink has valid fitness value

        times, y_output = self.individuals[individual_i].evaluate(final_t)
        self.individuals[individual_i].fitness(self.target_signal, times, y_output, fitness_type)

        i = individual_i
        while not in_order(i, self.individuals):

            # assume one side is ordered
            # move item to the unordered side, if ordered, then assume array is ordered, so leave loop.
            # sinking should be uni-directional

            if i + 1 < self.pop_size and not self.individuals[i].last_fitness <= self.individuals[i + 1].last_fitness:
                temp = self.individuals[i]
                self.individuals[i] = self.individuals[i + 1]
                self.individuals[i + 1] = temp
                i += 1
            elif i - 1 >= 0 and not self.individuals[i].last_fitness >= self.individuals[i - 1].last_fitness:
                temp = self.individuals[i]
                self.individuals[i] = self.individuals[i - 1]
                self.individuals[i - 1] = temp
                i -= 1
            else:
                break

    def lower_third_reproduction(self, cross_over_type="microbial"):
        """ Performs a round of reproduction. The lower third is replaced with off-spring from the top third."""

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

            self.individuals[idx] = parent1.cross_over(parent2, cross_over_type, 0.6)

    def weakest_individual_reproduction(self, cross_over_type: str = "microbial"):
        """ Replaces the weakest individual with the individual returned from cross-over. Fitness type specifies
            which heuristic is used to determine an individual's fitness"""

        best_individual = self.individuals[-1]
        weakest_individual = self.individuals[0]

        self.individuals[0] = best_individual.cross_over(weakest_individual, cross_over_type, 0.6)
        self.individuals[0].params.eval_valid = False

    def copy(self):
        copied_individuals = [individual.copy() for individual in self.individuals]
        environment = Deme(self.target_signal, self.struct, self.network_type, self.pop_size)
        environment.fill_individuals(individuals=copied_individuals)

        return environment

    def save_state(self, environment_index, state_file="state_file"):
        import os

        whatami = ''

        if isinstance(self.individuals[0], CTRNN):
            whatami = 'ctrnn'
        elif isinstance(self.individuals[0], KuramotoOscillator):
            whatami = 'kuramoto'

        state_file = f"{whatami}_{state_file}_{environment_index}"
        some_network = self.individuals[0]

        num_nodes = some_network.params.num_nodes
        signal_as_string = str(self.target_signal)
        target_type = self.target_signal.type
        audio_note = self.target_signal.audio_handler.note_name
        distribution = self.struct.distribution
        output_type = some_network.params.output_handler.method.strip()
        connection_type = self.struct.connection_type.strip()
        forcing_mask_type = self.struct.forcing_mask_type.strip()
        distribution_type = distribution.get_type()

        connection_string = ''
        for entry in self.struct.connection_array:
            i, j = entry
            connection_string += f"{i},{j} "

        state_string = f"{self.pop_size}\n{num_nodes}\n{connection_type}\n"\
                       f"{forcing_mask_type}\n{output_type}\n{distribution_type}\n{target_type}\n{audio_note}\n"\
                       f"{signal_as_string}\n{connection_string}\n{distribution}\n"

        for individual in self.individuals:
            eval_valid = individual.params.eval_valid
            last_fitness = individual.last_fitness
            genome = ''
            for gene in individual.params.genome:
                genome += f'{gene} '

            state_string += f"{eval_valid} {last_fitness} {genome}\n"

        if not os.path.exists("Saved States"):
            os.mkdir("Saved States")

        with open(f"Saved States/{state_file}", "w") as write_file:
            write_file.write(state_string)

    @staticmethod
    def load_deme(environment_index, forcing_signals, network_type, state_file="state_file"):

        state_file = f"{network_type}_{state_file}_{environment_index}"

        with open(f"Saved States/{state_file}") as read_file:
            state_string = read_file.readlines()

        pop_size = int(state_string[0])
        num_nodes = int(state_string[1])
        connection_type = state_string[2]
        forcing_mask_type = state_string[3]
        output_type = state_string[4]
        distribution_type = state_string[5].strip()
        target_type = state_string[6]
        audio_note = state_string[7].strip()
        signal_as_string = state_string[8]
        connection_string = state_string[9]
        distribution_string = state_string[10]
        target_signal = TargetSignal(0, 0.85, type=target_type)
        output_handler = OutputHandler(output_type)

        connection_array = []
        for entry in connection_string.split():
            i, j = entry.split(',')
            connection_array.append((int(i), int(j)))

        if distribution_type == "uniform":
            distribution_parameters = [[], []]
        else:
            distribution_parameters = []

        for entry in distribution_string.split():
            p1, p2 = entry.split(',')
            p1 = float(p1)
            p2 = float(p2)
            if distribution_type == "uniform":
                distribution_parameters[0].append(p1)
                distribution_parameters[1].append(p2)
            else:
                distribution_parameters.extend([p1, p2])

        distribution = Distribution.make_distribution(f"{network_type}_{distribution_type}")

        individuals = []
        for individual_specs in state_string[11:]:
            individual_specs = individual_specs.strip().split()
            fitness_valid = individual_specs[0]
            last_fitness = individual_specs[1]
            genome = [float(gene) for gene in individual_specs[2:]]
            network = None
            if network_type == 'ctrnn':
                ctrnn_params = CTRNNParameters(genome, output_handler, forcing_signals, connection_array)
                network = CTRNN(ctrnn_params)
            elif network_type == 'kuramoto':
                kuramoto_params = KuramotoParameters(genome, output_handler, forcing_signals, len(connection_array))
                network = KuramotoOscillator(kuramoto_params, connection_array)

            network.last_fitness = float(last_fitness)
            network.fitness_valid = bool(fitness_valid)
            individuals.append(network)
        
        struct = None
        if network_type == 'kuramoto':
            struct = KuramotoStructure(distribution, num_nodes, forcing_mask_type, connection_type, connection_array)
        elif network_type == 'ctrnn':
            struct = CTRNNStructure(distribution, num_nodes, forcing_mask_type, connection_type, connection_array)

        environment = Deme(target_signal, struct, network_type, pop_size)
        environment.fill_individuals(individuals=individuals)

        return environment



