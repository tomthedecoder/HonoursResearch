
if __name__ == "__main__":
    """ Set up and run"""

    from variables import *

    def ts(t):
        return np.sin(2*t)

    target_signal = lambda t: ts(t)

    ###########################
    # Initialize environments #
    ###########################

    load = False
    if load:
        for _ in range(num_runs):
            demes.append([])
            for gen_i in range(num_demes):
                environment = Environment.load_environment(gen_i, forcing_signals)
                environment.target_signal = target_signal
                demes[-1].append(environment)
    else:
        distribution = Distribution.make_distribution(distribution_type)
        output_handler = OutputHandler(handler_type)
        for _ in range(num_runs):
            demes.append([])
            ctrnn_structure = CTRNNStructure(distribution, num_nodes, connection_type=connection_type, center_crossing=True)
            for gen_i in range(num_demes):
                environment = Environment(target_signal, ctrnn_structure, pop_size=pop_size, mutation_chance=mutation_chance)
                environment.rank(final_t, fitness_type)
                environment.fill_individuals(output_handler, forcing_signals)
                demes[-1].append(environment)

    stdout.write("environments have been loaded, beginning run\n")

    #######################
    # Run the simulations #
    #######################

    # stores best ctrnn across all environments in demes
    best_environment = demes[0][-1]
    best_environment_id = 0
    best_ctrnn = best_environment.individuals[-1]

    best_run = 0

    best_fitness = []
    average_fitness = [[[] for _ in range(num_demes)] for _ in range(num_runs)]
    generations = []

    for run_id in range(num_runs):
        best_fitness.append([])
        generations.append([])

        generation_start_time = time()

        for gen_i in range(num_generations):
            generation_average_fitness = [0.0 for _ in range(num_demes)]

            # run algorithm on each environment
            for enviro_i, environment in enumerate(demes[run_id]):
                # chance to mutate individual which is not the current best in it's environment
                mc, cc = np.random.uniform(0, 1, 2)
                if mc >= 1 - environment.mutation_chance:
                    environment.mutate()

                # reproduction call
                environment.weakest_individual_reproduction(final_t, cross_over_type, fitness_type)

                # chance to breed with another individual from different environment
                last_strongest = environment.individuals[-1]
                environment.rank(final_t, "simpsons")
                if cc >= 1 - cross_over_probability:
                    environment = demes[run_id][np.random.randint(0, num_demes)]
                    j = np.random.randint(0, environment.pop_size-1)
                    individual = environment.individuals[j]
                    environment.individuals[j] = last_strongest.cross_over(individual, cross_over_type, 0.5)

                # determines best individual out of all environments
                if best_ctrnn.last_fitness > last_strongest.last_fitness:
                    best_environment_id = enviro_i
                    best_environment = environment
                    best_ctrnn = last_strongest
                    best_run = run_id

                # cumulative fitness across environment for this generation
                for individual in environment.individuals:
                    generation_average_fitness[enviro_i] += abs(individual.last_fitness) if individual.fitness_valid else 0

            # average the cumulative fitness
            for enviro_i, fitness in enumerate(generation_average_fitness):
                average_fitness[run_id][enviro_i].append(-fitness / pop_size)

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

    stdout.write(f"best ctrnn has fitness {-best_ctrnn.last_fitness}\nbest ctrnn weights ")

    for weight in best_ctrnn.params.weights:
        stdout.write(f"{weight} ")

    stdout.write(f"\nbest ctrnn taus {best_ctrnn.params.taus}\n")
    stdout.write(f"best ctrnn biases {best_ctrnn.params.biases}\n")
    stdout.write(f"best ctrnn forcing weights {best_ctrnn.params.forcing_weights}\n")

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
        plt.plot(generations, average_fitness[best_run][best_environment_id], color='g')
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











