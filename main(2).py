from Environment import *
import numpy as np
import matplotlib.pyplot as plt
from os import path
from sys import stdout
from scipy.fft import fft, fftfreq
from plot_all_neurons import plot_all_neurons
from genome_distribution import genome_distribution

# runs the GA


def target_signal(t):
    value = np.exp(-0.05 * t) * np.cos(t)

    p = np.random.randint(0, 1)
    thershold = 0.1
    spike_value = 1000
    if p >= thershold:
        return value + spike_value
    else:
        return value


def display_output():
    """ Set up and run"""

    # signal to interpolate
    environment = Environment(target_signal=target_signal, pop_size=3, mutation_chance=0.05)
    num_nodes = 1

    #connectivity_mask = [[1.0,0.0,0.0,1.0],
    #                     [1.0,1.0,0.0,1.0],
    #                     [0.0,1.0,1.0,1.0],
    #                     [0.0,0.0,0.0,1.0]]

    #connectivity_mask = [[1.0, 1.0],
    #                     [0, 1.0]]

    #connectivity_mask = None

    #connectivity_mask = [[1.0, 1.0, 1.0],
    #                     [1.0, 1.0, 1.0],
    #                     [1.0, 1.0, 1.0]]

    #individuals = [1, 2, 3]

    #connection_matrix = [[0.0, 0.0, 1.0, 0.0],
    #                     [0.0, 0.0, 1.0, 0.0],
    #                     [1.0, 1.0, 1.0, 1.0],
    #                     [0.0, 0.0, 1.0, 0.0]]

    #connection_matrix = [[0.0, 0.0, 0.0, 1.0, 0.0],
    #                     [0.0, 0.0, 0.0, 1.0, 0.0],
    #                     [0.0, 0.0, 0.0, 1.0, 0.0],
    #                     [1.0, 1.0, 1.0, 1.0, 1.0],
    #                     [0.0, 0.0, 0.0, 1.0, 0.0]]

    environment.fill_individuals(num_nodes, connection_matrix=None)

    final_t = np.ceil(12*np.pi)
    iterations = 0
    termination_threshold = np.Inf

    errors = []
    generations = []

    for i in range(iterations):
        mp = np.random.randint(0, 1)

        if 1 - mp >= environment.mutation_chance:
            environment.mutate()

        stdout.write(str(i) + " ")
        environment.weakest_individual_reproduction(final_t, cross_over_type="microbial", fitness_type="simpsons")

        errors.append(1 - environment.individuals[-1].simpsons_fitness(target_signal=target_signal, final_t=final_t))
        generations.append(i+1)

        """# if the worst individual is sufficiently bad, replace it with a new randomly generated one
        weakest_individual = environment.individuals[0]
        old_connection_matrix = weakest_individual.ctrnn.connectivity_mask
        fitness = weakest_individual.fitness(target_signal, final_t, fitness_type="simpsons")
        if fitness >= termination_threshold or fitness == np.nan:
            new_genome = environment.make_genome(weakest_individual.num_nodes)
            environment.individuals[0] = Individual(new_genome, weakest_individual.num_nodes, old_connection_matrix)"""

    environment.rank(final_t, "simpsons")
    best_individual = environment.individuals[-1]

    times = []
    y_target = []
    DT = 0.001

    best_individual.last_time = DT
    best_individual.ctrnn.node_values = np.array([0.0 for _ in range(best_individual.num_nodes)])
    best_individual.ctrnn.history = [[] for _ in range(best_individual.num_nodes)]
    best_individual.evaluate(reset=False, final_t=final_t, DT=DT)

    y_output = best_individual.ctrnn.history[-1]

    for idx in range(int(final_t/DT) - 1):
        times.append(DT * idx)
        y_target.append(environment.target_signal(times[-1]))

    plt.figure()
    plt.grid()
    plt.plot(times, y_target, 'b')
    plt.plot(times, y_output, 'g')
    plt.legend(["Target", "Output"])
    plt.title("CTRNN And Curve")
    plt.xlabel("Time(t)")
    plt.ylabel("Output(y)")

    # calls plt.show() for all the above plots
    plot_all_neurons(best_individual, final_t=final_t, step_size=DT)


display_output()