from Environment import *
import numpy as np
import matplotlib.pyplot as plt
from os import path
from sys import stdout
#from scipy.fft import fft, fftfreq
from plot_all_neurons import plot_all_neurons
from genome_distribution import genome_distribution


if __name__ == "__main__":
    """ Set up and run"""

    # signal to interpolate
    target_signal = lambda t: np.exp(-0.05 * t) * np.cos(t)
    load = False
    if path.exists("state_file") and load:
        environment = Environment.load_environment("state_file")
        for individual in environment.individuals:
            print(individual.fitness(target_signal, fitness_type="simpsons", final_t=np.ceil(12*np.pi)), individual.genome)
        print()
        environment.target_signal = target_signal
        environment.mutation_chance = 0.05
    else:
        environment = Environment(target_signal=target_signal, pop_size=1, mutation_chance=0.05)
        num_nodes = 2
        connectivity_array = [(1, 1), (1, 2), (2, 2)]
        #connectivity_array = None

        center_crossing = True

        environment.fill_individuals(num_nodes=num_nodes, connection_array=connectivity_array)

    final_t = np.ceil(12*np.pi)
    iterations = 0

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

    environment.rank(final_t, "simpsons")
    best_individual = environment.individuals[-1]
    times = []
    y_target = []
    DT = 0.01

    best_individual.last_time = 0
    best_individual.ctrnn.node_values = np.array([0.0 for _ in range(best_individual.num_nodes)])
    best_individual.ctrnn.history = [[] for _ in range(best_individual.num_nodes)]
    best_individual.save = True
    best_individual.ctrnn.step_size = 0.01
    best_individual.evaluate(final_t=final_t)

    print(best_individual.genome)

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

    """fourier_of_output = fft(y_output)
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











