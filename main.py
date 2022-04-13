
if __name__ == "__main__":
    """ Set up and run"""

    from variables import *

    ####################
    # Initialize Demes #
    ####################

    run_holder = RunHolder(num_demes, num_runs, num_networks, cross_over_type, fitness_type)
    target_signal = TargetSignal(start_t, final_t, 'audio', lambda t: np.sin(2 * t))
    result_string = ''

    load = False
    if load:

        # fetch demes for both networks from stored files

        spawn = "load"
        for net_i, network in enumerate(network_types):
            for run_i in range(num_runs):
                for gen_i in range(num_demes):
                    deme = Deme.load_deme(gen_i, forcing_signals, network)
                    deme.target_signal = target_signal
                    run_holder.set(net_i, run_i, gen_i, deme)
    else:
        spawn = "created"
        output_handler = OutputHandler(handler_type)
        for net_i, network in enumerate(network_types):
            distribution = Distribution.make_distribution(f'{network}_{distribution_type}')

            if network == "ctrnn":
                structure = CTRNNStructure(distribution, num_nodes, connection_type, center_crossing=True)
            elif network == "kuramoto":
                structure = KuramotoStructure(distribution, num_nodes, connection_type)
            else:
                raise ValueError("Invalid network type")

            for run_i in range(num_runs):
                for deme_i in range(num_demes):

                    #  create demes

                    if run_i == 0 or True:
                        deme = Deme(target_signal, structure, network, pop_size=pop_size)
                        deme.fill_individuals(output_handler, forcing_signals)
                        deme.rank(final_t, fitness_type)
                        run_holder.add(net_i, run_i, deme)

                    # copy first demes created previously for the first run/

                    else:
                        deme_to_copy = run_holder.get(net_i, 0, deme_i)
                        copied_deme = deme_to_copy.copy()
                        run_holder.add(net_i, run_i, copied_deme)

    # up-dates best network

    run_holder.init_best()

    stdout.write(f"environments have been {spawn} and ranked, beginning run\n")
    result_string += f"creation: {spawn}\ndistribution: {distribution_type}\ncross-over: {cross_over_type}" \
                     f"\ncross over probability: {cross_over_probability}\nfitness function: {fitness_type}\n" \
                     f"output handler: {handler_type}\ntopology: {connection_type}\nforcing mask: {mask_type}\n" \
                     f"number of nodes: {num_nodes}\npopulation size: {pop_size}\ngeneration number: {num_generations}\n" \
                     f"number of runs: {num_runs}\nnumber of demes: {num_demes}\n" \
                     f"Target signal: {target_signal.type}, started at {start_t} and ended at {final_t}"

    #######################
    # Run the simulations #
    #######################

    # run the algorithm for each network

    for net_i in range(num_networks):

        # time it takes to do all runs

        generation_start_time = time()

        # the first layer of the algorithm is the meta-layer, repeats actual genetic algorithm num_run times

        for run_i in range(num_runs):

            # run a simulation of the genetic algorithm

            for gen_i in range(num_generations):
                if gen_i % 100 == 0:
                    print(gen_i)
                generation_average_fitness = [0.0 for _ in range(num_demes)]

                # run algorithm on each deme

                for deme_i, deme in enumerate(run_holder.get_demes(net_i, run_i)):

                    # mutate individual, not the best one, then re-evaluate

                    deme.sink(deme.mutate(), final_t, fitness_type)

                    # rank based selection method, replace the weakest individual cross_over(...) -> new individual

                    deme.weakest_individual_reproduction(cross_over_type)

                    # re-evaluate

                    deme.sink(0, final_t, fitness_type)

                    # chance to perform inter-deme cross-over
                    # will place individual in correct location afterwards

                    if np.random.uniform(0, 1, 1) >= 1 - cross_over_probability:
                        D, i = run_holder.get_demes(net_i, run_i).genetic_drift(deme_i, final_t)
                        D.sink(i, final_t, fitness_type)

                    # determines best network out of all environments

                    if not run_holder.best_networks[net_i].last_fitness >= deme.individuals[-1].last_fitness:
                        run_holder.best_deme_ids[net_i] = deme_i
                        run_holder.best_demes[net_i] = deme
                        run_holder.best_networks[net_i] = deme.individuals[-1]
                        run_holder.best_runs[net_i] = run_i

                    # cumulative fitness across environment for this generation

                    for individual in deme.individuals:
                        generation_average_fitness[deme_i] += individual.last_fitness

                # average the cumulative fitness

                for deme_i, fitness in enumerate(generation_average_fitness):
                    run_holder.average_fitness[net_i][run_i][deme_i].append(fitness / pop_size)

                run_holder.best_fitness[net_i][run_i].append(run_holder.best_networks[net_i].last_fitness)

            stdout.write(f"{run_i+1} of {network_types[net_i]}\n")
        generation_end_time = time()
        s = f"\nRun finished for the {network_types[net_i]} model finished, runtime is {generation_end_time - generation_start_time} seconds.\n"
        stdout.write(s)
        result_string += s

    # write results to ostream

    s = "\nResult Table\n" + 50*"-" + "\n"

    stdout.write(s)
    result_string += s

    for net_i in range(num_networks):
        s = f"{network_types[net_i]} model\nbest fitness: {run_holder.best_networks[net_i].last_fitness}\n{run_holder.best_networks[net_i].params}\n" + 50 * "-" + "\n"

        stdout.write(s)
        result_string += s

    # save the state of best demes

    s = "\nsaving state .... "
    stdout.write(s)
    result_string += s

    for net_i, run_i in enumerate(run_holder.best_runs):
        topology = run_holder.demes[net_i][run_i]
        for i, deme in enumerate(topology):
             deme.save_state(i)

    s = "state saved\n"
    stdout.write(s)
    result_string += s

    ########################
    # Plot best network(s) #
    ########################

    # plot genomes of environments which contain the best network
    # for ni, network in enumerate(network_types):
    #    box_plot(network, run_holder.best_demes[ni])

    # plot output of last neuron
    plt.figure(figsize=(18, 18), dpi=100)
    plt.title("Target Curve And Network Outputs", fontsize=20)
    times = []
    legend = []
    for network in run_holder.best_networks:
        network.reset()
        times, y = network.evaluate(final_t)
        if isinstance(network, KuramotoOscillator):
            legend.append("Kuramoto Oscillator")
            plt.plot(times, y)
        elif isinstance(network, CTRNN):
            legend.append("CTRNN")
            plt.plot(times, y)
    plt.grid()
    legend.append("Target Curve")
    plt.plot(times, target_signal(times))
    plt.legend(legend, fontsize=18)

    if num_nodes > 0:
        for i, network in enumerate(network_types):
            plot_all_nodes(run_holder.best_networks[i], final_t)

    # fitness plots for each network
    if num_generations > 2:
        generations = range(num_generations)
        for ni, network in enumerate(network_types):
            di = run_holder.best_deme_ids[ni]
            ri = run_holder.best_runs[ni]
            new_name = "Kuramoto Oscillator" if network == "kuramoto" else "CTRNN"
            plt.figure(figsize=(12, 12), dpi=100)
            plt.grid()
            plt.title(f"Average Fitness For {new_name}", fontsize=20)
            plt.xlabel("Generation", fontsize=20)
            plt.ylabel("Fitness", fontsize=20)
            plt.plot(generations, run_holder.average_fitness[ni][ri][di])
            plt.figure(figsize=(12, 12), dpi=100)
            plt.grid()
            plt.title(f"Best Fitness For {new_name}", fontsize=20)
            plt.xlabel("Generation", fontsize=20)
            plt.ylabel("Fitness", fontsize=20)
            plt.plot(generations, run_holder.best_fitness[ni][ri])

    i = 1
    while os.path.exists(f"Results/run_result{i}"):
        i += 1

    if not os.path.exists("Results"):
        os.mkdir("Results")

    with open(f"Results/run_result{i}", 'w+') as write_handle:
        write_handle.write(result_string)

    plt.show()










