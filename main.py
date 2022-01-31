from Environment import Environment
from Distribution import Distribution
from plot_all_neurons import plot_all_neurons
from genome_distribution import *
from sys import stdout
from time import time
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    """ Set up and run"""

    def ts(t):
        return np.cos(t)

    target_signal = lambda t: ts(t)

    #################################################
    # Set up parameters and initialize environments #
    #################################################

    # probability that cross over will occur between demes
    cross_over_probability = 0.5
    num_demes = 4
    num_nodes = 4
    connectivity_array = None
    demes = []

    final_t = np.ceil(12*np.pi)
    fitness_type = "simpsons"
    cross_over_type = "microbial"
    distribution_type = "uniform"
    num_generations = 200
    num_runs = 5

    lows, highs = uniform_parameters()
    lambdas = poisson_parameters()
    mus, stds = normal_parameters()

    load = False
    if load:
        for _ in range(num_runs):
            demes.append([])
            for i in range(num_demes):
                environment = Environment.load_environment(i)
                environment.target_signal = target_signal
                demes[-1].append(environment)
    else:
        for _ in range(num_runs):
            demes.append([])
            for i in range(num_demes):
                distribution = Distribution.make_distribution(distribution_type)
                environment = Environment(target_signal, distribution, 200, False, 0.90)
                environment.rank(final_t, fitness_type)
                environment.fill_individuals(num_nodes, connectivity_array)
                demes[-1].append(environment)

    stdout.write("environments have been loaded, beginning run\n")

    #######################
    # Run the simulations #
    #######################

    # stores best ctrnn across all environments in demes
    best_environment = demes[0][-1]
    best_ctrnn = best_environment.individuals[-1]

    best_run = 0

    best_fitness = []
    average_fitness = []
    generations = []

    for run_id in range(num_runs):
        best_fitness.append([])
        average_fitness.append([])
        generations.append([])

        generation_start_time = time()

        for i in range(num_generations):
            # average fitness is the average across all demes
            generation_average_fitness = 0

            # run algorithm on each environment
            for environment_index, environment in enumerate(demes[run_id]):
                mc, cc = np.random.uniform(0, 1, 2)
                if mc >= 1 - environment.mutation_chance:
                    environment.mutate()

                environment.weakest_individual_reproduction(final_t, cross_over_type, fitness_type)

                last_strongest = environment.individuals[-1]
                if cc >= 1 - cross_over_probability:
                    environment = demes[run_id][np.random.randint(0, num_demes)]
                    j = np.random.randint(0, environment.pop_size-1)
                    individual = environment.individuals[j]
                    environment.individuals[j] = last_strongest.cross_over(individual, cross_over_type, 0.5)

                if best_ctrnn.last_fitness > last_strongest.last_fitness:
                    best_environment = environment
                    best_ctrnn = last_strongest
                    best_run = run_id

                #generation_average_fitness = 0
                #for individual in environment.individuals:
                #    generation_average_fitness += individual.last_fitness
                #generation_average_fitness /= 1/environment.pop_size

            average_fitness[-1].append(-generation_average_fitness)
            best_fitness[-1].append(-best_ctrnn.last_fitness)

        generation_end_time = time()
        stdout.write(f"run finished for {num_generations} generations, runtime is {generation_end_time - generation_start_time} seconds."
                     f" The best Fitness level is {best_ctrnn.last_fitness}\n")

    # save the state of best demes
    stdout.write(f"all runs complete. The best run was {best_run+1}\n")
    stdout.write("saving state\n")
    for index, environment in enumerate(demes[best_run]):
        environment.save_state(index)
    stdout.write("state has been saved\n")

    stdout.write(f"best ctrnn has fitness {best_ctrnn.last_fitness}\nbest ctrnn weights ")

    for weight in best_ctrnn.weights:
        stdout.write(f"{weight} ")

    stdout.write(f"\nbest ctrnn taus {best_ctrnn.taus}\n")
    stdout.write(f"best ctrnn biases {best_ctrnn.biases}\n")
    stdout.write(f"best ctrnn forcing weights {best_ctrnn.forcing_weights}\n")
    stdout.write(f"best ctrnn shift {best_ctrnn.shift}\n")

    ###################
    # Plot best ctrnn #
    ###################

    best_ctrnn.reset()

    y_target = []
    times, y_output = best_ctrnn.evaluate(final_t=final_t)
    for t in times:
        y_target.append(target_signal(t))

    if num_nodes > 1:
        plot_all_neurons(best_ctrnn, final_t)

    box_plot(best_environment)
    #plot_distribution(best_environment)
    if num_generations > 2:
        generations = [i+1 for i in range(num_generations)]

        plt.figure()
        plt.grid()
        plt.plot(generations, best_fitness[best_run], 'b')
        plt.title('Fitness Of Best Individual')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')

        plt.figure()
        plt.grid()
        plt.plot(generations, average_fitness[best_run], 'g')
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











