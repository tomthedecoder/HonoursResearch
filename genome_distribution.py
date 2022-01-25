import matplotlib.pyplot as plt
from Distribution import *


def box_plot(environment):
    """ Reads from the state file and displays a distribution of genomes"""

    num_nodes = environment.individuals[0].num_nodes
    num_weights = environment.individuals[0].num_weights

    weight_counts = []
    biases_counts = []
    taus_counts = []
    input_weight_counts = []

    for individual in environment.individuals:
        genome = individual.genome
        for idx, gene in enumerate(genome):
            # find proper count array
            if idx <= num_weights-1:
                weight_counts.append(gene)
            elif idx <= num_weights + num_nodes-1:
                taus_counts.append(gene)
            elif idx <= num_weights + 2 * num_nodes-1:
                biases_counts.append(gene)
            else:
                input_weight_counts.append(gene)

    fig1, axs1 = plt.subplots(2)
    fig2, axs2 = plt.subplots(2)

    axs1[0].set_title("Weights")
    axs1[1].set_title("Biases")
    axs2[0].set_title("Taus")
    axs2[1].set_title("Input Weights")

    axs1[0].boxplot(weight_counts, vert=False)
    axs1[1].boxplot(biases_counts, vert=False)
    axs2[0].boxplot(taus_counts, vert=False)
    axs2[1].boxplot(input_weight_counts, vert=False)


def plot_distribution(environment):
    """ plots genomes in a distributions style"""

    def add_to_counts(parameter, count_array, partition_distance):
        """ adds the parameter to the correct count slot"""

        placement = int(parameter * partition_distance)

        # search count array for parameter
        found = False
        for idx, pos, count in enumerate(count_array):
            if placement == pos:
                count_array[idx] = count + 1
                found = True
                break
        if not found:
            count_array.append([placement, 1])

    def scale_array(count_array, total):
        """ takes a count array and returns a ratio array"""

        placements = []
        ratios = []

        for placement, count in count_array:
            placements.append(placement)
            ratios.append(count/total)

        return placements, ratios

    partition_distance = 0.05

    weight_counts = []
    tau_counts = []
    bias_counts = []
    input_weight_counts = []

    # get count arrays
    for ctrnn in environment.individuals:
        for weight in ctrnn.weights:
            add_to_counts(weight.value, weight_counts, partition_distance)
        for tau in ctrnn.taus:
            add_to_counts(tau_counts, tau, partition_distance)
        for bias in ctrnn.biases:
            add_to_counts(bias_counts, bias, partition_distance)
        for input_weights in ctrnn.input_weights:
            add_to_counts(input_weight_counts, input_weights, partition_distance)

    # ratio arrays
    total_weights = environment.pop_size * environment.individuals[0].num_weights
    total_num = environment.pop_size * environment.individuals[0].num_nodes

    weight_placements, weight_ratios = scale_array(weight_counts, total_weights)
    tau_placements, tau_ratios = scale_array(tau_counts, total_num)
    bias_placements, bias_ratios = scale_array(bias_counts, total_num)
    input_weight_placements, input_weight_ratios = scale_array(input_weight_counts, total_num)

    plt.figure()
    plt.plot(weight_placements, weight_ratios)
    plt.plot(weight_placements, weight_ratios, 'o')