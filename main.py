from Environment import *
from Distribution import Distribution
from plot_all_neurons import plot_all_neurons
from genome_distribution import genome_distribution
from sys import stdout
from time import time
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    """ Set up and run"""

    def ts(t):
        return np.sin(t)

    target_signal = lambda t: ts(t)

    # probability that cross over will occur between demes
    cross_over_probability = 0.75
    num_demes = 2
    demes = []

    num_nodes = 1
    connectivity_array = None

    load = False
    if load:
        for i in range(num_demes):
            environment = Environment.load_environment(i)
            environment.target_signal = target_signal
            demes.append(environment)
    else:
        for _ in range(num_demes):
            environment = Environment(target_signal=target_signal, pop_size=50, mutation_chance=0.90, center_crossing=False)
            environment.fill_individuals(num_nodes=num_nodes, connection_array=connectivity_array)
            demes.append(environment)

    stdout.write("environments have been loaded, beginning run\n")

    final_t = np.ceil(12*np.pi)
    num_generations = 50
    num_iterations = 10

    best_fitness = []
    average_fitness = []
    generations = []

    start_time = time()


    for i in range(num_generations):
        # average fitness is the average across all demes
        generation_average_fitness = 0
        strongest_individual_fitness = np.Infinity

        for environment_index, environment in enumerate(demes):
            mutation_chance, cross_over_chance = np.random.uniform(0, 1, 2)
            if mutation_chance >= 1 - environment.mutation_chance:
                environment.mutate()

            last_strongest = environment.individuals[-1]
            if cross_over_chance >= 1 - cross_over_probability and False:
                environment = demes[np.random.randint(0, num_demes)]
                j = np.random.randint(1, environment.pop_size)
                individual = environment.individuals[j]
                environment.individuals[j] = last_strongest.cross_over(individual, "microbial", 0.5)

            environment.weakest_individual_reproduction(final_t, cross_over_type="microbial", fitness_type="simpsons")

            if last_strongest != environment.individuals[-1] and i > 0:
                stdout.write(f"change in strongest of environment {environment_index} has occurred at iteration {i}\n")

            if strongest_individual_fitness > last_strongest.last_fitness:
                strongest_individual_fitness = last_strongest.last_fitness

            generation_average_fitness = 0
            for individual in environment.individuals:
                generation_average_fitness += individual.last_fitness
            generation_average_fitness /= 1/environment.pop_size

        best_fitness.append(-strongest_individual_fitness)

        generation_average_fitness /= 1/num_demes
        average_fitness.append(-generation_average_fitness)

        generations.append(i+1)

    end_time = time()
    stdout.write(f"run finished for {num_generations} iterations, runtime is {end_time - start_time} seconds\n")

    stdout.write("ranking individuals\n")
    # perform rank on individuals across all demes
    for environment in demes:
        environment.rank(final_t, "simpsons")
    stdout.write("individuals ranked\n")

    # save the state of demes
    stdout.write("saving state\n")
    for index, environment in enumerate(demes):
        environment.save_state(index)
    stdout.write("state has been saved\n")

    # get best ctrnn across all environments in demes
    best_environment = demes[0]
    best_ctrnn = best_environment.individuals[-1]
    strongest_individual_fitness = best_ctrnn.last_fitness
    for environment in demes:
        local_best_ctrnn = environment.individuals[-1]
        if strongest_individual_fitness > local_best_ctrnn.last_fitness:
            best_environment = environment
            strongest_individual_fitness = local_best_ctrnn.last_fitness
            best_ctrnn = local_best_ctrnn

    stdout.write(f"best ctrnn has fitness {best_ctrnn.last_fitness} and genome {best_ctrnn.genome}\n")

    ###################
    # Plot best ctrnn #
    ###################
    best_ctrnn.reset()

    times = []
    y_target = []
    DT = 0.01

    best_ctrnn.step_size = DT
    y_output = best_ctrnn.evaluate(final_t=final_t)

    for idx in range(int(final_t/DT)):
        times.append(DT * idx)
        y_target.append(target_signal(times[-1]))

    if num_nodes > 1:
        plot_all_neurons(best_ctrnn, final_t=final_t, step_size=DT)

    genome_distribution(best_environment)
    if num_generations > 0:
        plt.figure()
        plt.grid()
        plt.plot(generations, best_fitness, 'b')
        plt.title('Fitness Of Best Individual')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')

        plt.figure()
        plt.grid()
        plt.plot(generations, average_fitness, 'g')
        plt.title('Fitness Of Average Individual')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')

    plt.figure()
    plt.grid()
    plt.plot(times, y_target, 'b')
    plt.plot(times, y_output, 'g')
    plt.legend(["Target", "Output"])
    plt.title("CTRNN And Curve")
    plt.xlabel("Time(t)")
    plt.ylabel("Output(y)")

    plt.show()











