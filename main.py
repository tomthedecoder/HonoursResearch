from Environment import *
import numpy as np
import matplotlib.pyplot as plt
from os import path
from sys import stdout
from scipy import fft
from plot_all_neurons import plot_all_neurons
from genome_distribution import genome_distribution
from time import time


def T(t):
    return -np.power(t, 2)


if __name__ == "__main__":
    """ Set up and run"""

    target_signal = lambda t: np.sin(t)
    load = False
    if path.exists("state_file") and load:
        environment = Environment.load_environment("state_file")
        environment.target_signal = target_signal
    else:
        environment = Environment(target_signal=target_signal, pop_size=200, mutation_chance=0.05, center_crossing=True)
        num_nodes = 1
        #connectivity_array = [(1, 1), (1, 2), (2, 2)]
        connectivity_array = None

        environment.fill_individuals(num_nodes=num_nodes, connection_array=connectivity_array)

    final_t = np.ceil(12*np.pi)
    iterations = 1000

    errors = []
    generations = []

    start_time = time()

    for i in range(iterations):
        mp = np.random.uniform(0, 1, 1)[0]
        if mp >= 1 - environment.mutation_chance:
            environment.mutate()

        last_strongest = environment.individuals[-1]
        environment.weakest_individual_reproduction(final_t, cross_over_type="microbial", fitness_type="simpsons")

        if last_strongest != environment.individuals[-1] and i > 0:
            stdout.write("change in strongest has occurred at iteration {}\n".format(i))

        errors.append(environment.individuals[-1].last_fitness)
        generations.append(i+1)

    end_time = time()
    stdout.write("runtime is {} seconds\n".format(end_time - start_time))

    environment.rank(final_t, "simpsons")
    best_individual = environment.individuals[-1]
    best_individual.ctrnn.reset(True)

    times = []
    y_target = []
    DT = 0.01

    best_individual.ctrnn.save = True
    best_individual.ctrnn.step_size = DT
    best_individual.evaluate(final_t=final_t)

    worst_individual = environment.individuals[0]
    worst_individual.ctrnn.reset(True)
    worst_individual.ctrnn.save = True
    worst_individual.ctrnn.step_size = DT

    print(best_individual.genome)
    print(best_individual.last_fitness)

    y_output = best_individual.ctrnn.node_history[-1]
    for idx in range(int(final_t/DT)):
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

    plt.figure()
    plt.grid()
    plt.plot(generations, errors, 'b')
    plt.title('Fitness Of Best Individual')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')

    # number of sample points
    N = 800
    # spacing between points
    T = 1.0/N

    """"fourier_of_output = fft(y_output)
    fourier_of_target = fft(y_target)
    tf = fftfreq(N, T)[:N // 2]

    plt.figure()
    plt.title("Fourier Transforms")
    plt.plot(tf, 1.0 / N * np.abs(fourier_of_output[0:N // 2]), 'g')
    plt.plot(tf, 1.0 / N * np.abs(fourier_of_target[0:N // 2]), 'b')
    plt.legend(["Output", "Target"])
    plt.grid()"""

    # calls plt.show() for all the above plots
    plot_all_neurons(best_individual, final_t=final_t, step_size=DT)

    environment.save_state()

    # display genome distribution by reading from state file
    genome_distribution()

    contents = ""
    with open("best_individual", "w") as best_file:
        for idx, y in enumerate(best_individual.genome):
            contents += str(y) + " "
        best_file.write(contents)

    contents = ""
    with open("output_file", "w") as output_file:
        for idx, y in enumerate(y_output):
            contents += str(y) + " "
        output_file.write(contents)

    contents = ""
    with open("time_file", "w") as time_file:
        for idx, t in enumerate(times):
            contents += str(t) + " "
        time_file.write(contents)

    contents = ""
    with open("targets_file", "w") as targets_file:
        for idx, y in enumerate(y_target):
            contents += str(y) + " "
        targets_file.write(contents)











