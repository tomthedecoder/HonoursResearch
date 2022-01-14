import matplotlib.pyplot as plt


def genome_distribution(environment):
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

