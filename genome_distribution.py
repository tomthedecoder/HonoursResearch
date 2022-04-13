import matplotlib.pyplot as plt


def box_plot(network_name, environment):
    """ Reads from the state file and displays a distribution of genomes"""

    num_nodes = environment.individuals[0].num_nodes
    num_weights = environment.individuals[0].num_weights

    weight_counts = []
    biases_counts = []
    taus_counts = []
    input_weight_counts = []
    natural_frequencies_counts = []
    k_counts = []

    for individual in environment.individuals:
        genome = individual.params.genome
        for idx, gene in enumerate(genome):
            # find proper count array for network
            if environment.network_type == "ctrnn":
                if idx <= num_weights-1:
                    weight_counts.append(gene)
                elif idx <= num_weights + num_nodes-1:
                    taus_counts.append(gene)
                elif idx <= num_weights + 2 * num_nodes-1:
                    biases_counts.append(gene)
                else:
                    input_weight_counts.append(gene)
            elif environment.network_type == "kuramoto":
                if idx < num_nodes:
                    natural_frequencies_counts.append(gene)
                elif idx < 2 * num_nodes:
                    k_counts.append(gene)

    if network_name == "ctrnn":
        fig1, axs1 = plt.subplots(2, figsize=(12, 12), dpi=100)
        fig2, axs2 = plt.subplots(2, figsize=(12, 12), dpi=100)

        fig1.suptitle("CTRNN Parameters", fontsize=20)

        axs1[0].set_title("Weights", fontsize=18)
        axs1[1].set_title("Biases", fontsize=18)
        axs2[0].set_title("Taus", fontsize=18)
        axs2[1].set_title("Forcing Weights", fontsize=18)

        axs1[0].boxplot(weight_counts, vert=False)
        axs1[1].boxplot(biases_counts, vert=False)
        axs2[0].boxplot(taus_counts, vert=False)
        axs2[1].boxplot(input_weight_counts, vert=False)
    elif network_name == "kuramoto":
        fig1, axs1 = plt.subplots(2, figsize=(12, 12), dpi=100)

        fig1.suptitle("Kuramoto Oscillator Parameters", fontsize=20)

        axs1[0].set_title("Natural Frequencies", fontsize=18)
        axs1[1].set_title("Coupling Strength", fontsize=18)

        axs1[0].boxplot(natural_frequencies_counts, vert=False)
        axs1[1].boxplot(k_counts, vert=False)


def plot_distribution(environment):
    """ plots genomes in a distributions style"""

    def add_to_counts(parameter, count_array, partition_distance):
        """ adds the parameter to the correct count slot"""

        placement = int(parameter // partition_distance)

        # search count array for parameter
        found = False
        for idx, item in enumerate(count_array):
            pos, count = item
            if placement == pos:
                count_array[idx][1] = count + 1
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

    partition_distance = 0.1

    weight_counts = []
    tau_counts = []
    bias_counts = []
    forcing_weight_counts = []

    # get count arrays
    for ctrnn in environment.individuals:
        for weight in ctrnn.weights:
            add_to_counts(weight.value, weight_counts, partition_distance)
        for tau in ctrnn.taus:
            add_to_counts(tau, tau_counts, partition_distance)
        for bias in ctrnn.biases:
            add_to_counts(bias, bias_counts, partition_distance)
        for forcing_weight in ctrnn.forcing_weights:
            add_to_counts(forcing_weight, forcing_weight_counts, partition_distance)

    # ratio arrays
    total_weights = environment.pop_size * environment.individuals[0].num_weights
    total_num = environment.pop_size * environment.individuals[0].num_nodes

    weight_placements, weight_ratios = scale_array(weight_counts, total_weights)
    tau_placements, tau_ratios = scale_array(tau_counts, total_num)
    bias_placements, bias_ratios = scale_array(bias_counts, total_num)
    input_weight_placements, input_weight_ratios = scale_array(forcing_weight_counts, total_num)

    plt.figure()
    plt.title("Weight Distribution")
    plt.plot(weight_placements, weight_ratios, 'o')

    plt.figure()
    plt.title("Tau Distribution")
    plt.plot(tau_placements, tau_ratios, 'o')

    plt.figure()
    plt.title("Bias Distribution")
    plt.plot(bias_placements, bias_ratios, 'o')

    plt.figure()
    plt.title("Input Weights Distribution")
    plt.plot(input_weight_placements, input_weight_ratios, 'o')

    plt.show()


test = False
if test:
    from Deme import *

    environment = Deme.load_deme(1)
    plot_distribution(environment)