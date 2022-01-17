from Environment import Environment
from Distribution import Distribution
from plot_all_neurons import plot_all_neurons
from genome_distribution import genome_distribution
from sys import stdout
from time import time
import numpy as np
import matplotlib.pyplot as plt

# True 6.189763684283798 0.2888151902249554 0.3389540365826371 0.0014337711891763139 -0.9193451699551511 -0.9311019000065877 0.5971595357971116 0.9854802881753641 0.4526070780700566 0.8218117211634968 0.6170409845322968 0.7914348937737903 -0.8758224067915814 -0.3609363364306375 -0.0007035658009828527 0.6704859890347352 -0.220070362115518 0.6843016681104344 -0.03818007409973534 0.9557705192552659 -0.833428706529816 0.9202280019883518 0.5648972455310335 -0.8390453422502151 -0.8018667118528571 0.5112379303758579 1.4131278827204015 1.4786706617118477 1.5428651279497059 1.740358474249942 1.5535640439764604 -0.5255655513775392 -0.03771191527361606 0.7532472634396161 0.011421522104198445 0.4582009599520509 0.6668571852390509 0.39516787740502907 -0.6374476731715055 0.7215981919501118 -0.510573978179276


def get_lows_highs(num_demes):
    """ returns lows and highs arrays for distribution class based off maximum values of parameters"""

    tau_low = 0
    tau_high = 3
    bias_low = -5
    bias_high = 5
    weight_low = -5
    weight_high = 5
    input_weight_low = -4
    input_weight_high = 4

    lows = []
    highs = []

    for idx in range(num_demes):
        lows.append([])
        highs.append([])

        lows[-1].append(weight_low + idx * (weight_high - weight_low)/num_demes)
        lows[-1].append(tau_low + idx * (tau_high - tau_low)/num_demes)
        lows[-1].append(bias_low + idx * (bias_high - bias_low)/num_demes)
        lows[-1].append(input_weight_low + idx * (input_weight_high - input_weight_low)/num_demes)

        highs[-1].append(weight_low + (idx + 1) * (weight_high - weight_low)/num_demes)
        highs[-1].append(tau_low + (idx + 1) * (tau_high - tau_low)/num_demes)
        highs[-1].append(bias_low + (idx + 1) * (bias_high - bias_low)/num_demes)
        highs[-1].append(input_weight_low + (idx + 1) * (input_weight_high - input_weight_low)/num_demes)

    return lows, highs


if __name__ == "__main__":
    """ Set up and run"""

    def ts(t):
        return np.sin(2 * t)

    target_signal = lambda t: ts(t)

    #################################################
    # Set up parameters and initialize environments #
    #################################################

    # probability that cross over will occur between demes
    cross_over_probability = 0.5
    num_demes = 5
    num_nodes = 5
    connectivity_array = None
    demes = []

    final_t = np.ceil(3*np.pi)
    fitness_type = "simpsons"
    cross_over_type = "microbial"
    num_generations = 250
    num_runs = 4

    lows, highs = get_lows_highs(num_demes)

    load = True
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
                distribution = Distribution(lows[i], highs[i])
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
                mutation_chance, cross_over_chance = np.random.uniform(0, 1, 2)
                if mutation_chance >= 1 - environment.mutation_chance:
                    environment.mutate()

                environment.weakest_individual_reproduction(final_t, cross_over_type, fitness_type)

                last_strongest = environment.individuals[-1]
                if cross_over_chance >= 1 - cross_over_probability:
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











