import matplotlib.pyplot as plt
import numpy as np


def genome_distribution(filename="state_file"):
    """ Reads from the state file and displays a distribution of genomes"""

    with open(filename, "r") as read_file:
        contents = read_file.readlines()

    num_nodes = int(contents[1])

    weight_counts = []
    biases_counts = []
    taus_counts = []

    for ln, genome in enumerate(contents[3 + num_nodes:]):
        num = ""
        numbers_so_far = 0
        for idx, char in enumerate(genome):
            if char == " ":
                numbers_so_far += 1
                # find proper count dict
                if numbers_so_far <= num_nodes ** 2:
                    weight_counts.append(float(num))
                elif num_nodes ** 2 < numbers_so_far <= num_nodes ** 2 + num_nodes:
                    taus_counts.append(float(num))
                else:
                    biases_counts.append(float(num))

                # add count
                num = ""
            num += char

    fig, axs = plt.subplots(3)

    axs[0].set_title("Weights")
    axs[1].set_title("Biases")
    axs[2].set_title("Taus")

    axs[0].boxplot(weight_counts, vert=False)
    axs[1].boxplot(biases_counts, vert=False)
    axs[2].boxplot(taus_counts, vert=False)

    plt.show()
