from Individual import Individual
import numpy as np
import matplotlib.pyplot as plt


def plot_all_neurons(individual, final_t=10, step_size=0.01):
    """ Outputs a plot of each neuron"""

    if individual.num_nodes <= 0:
        raise ValueError("CTRNN must have at least one node")

    # reset ctrnn
    individual.last_time = 0
    individual.ctrnn.node_values = np.array([0.0 for _ in range(individual.num_nodes)])
    individual.ctrnn.history = [[] for _ in range(individual.num_nodes)]
    individual.save = True
    individual.ctrnn.step_size = step_size

    individual.evaluate(final_t=final_t)

    times = []
    for idx in range(int(final_t/step_size)):
        times.append(idx * step_size)

    if individual.num_nodes == 1:
        plt.figure()
        plt.title("Node 1")
        plt.grid()
        plt.xlabel("Time(t)")
        plt.ylabel("Output(y)")
        plt.plot(times, individual.ctrnn.node_history[0])
        plt.show()
        return

    # maximum of 5 nodes to one plot
    MAX_NODE = 5.0
    nodes_plotted = 0
    for idx in range(int(np.ceil(individual.num_nodes / MAX_NODE))):
        nodes_in_plot = min(individual.num_nodes, int(MAX_NODE))
        fig, axs = plt.subplots(nodes_in_plot)
        for node in range(nodes_in_plot):
            axs[node].plot(times, individual.ctrnn.node_history[node + nodes_plotted])
            axs[node].set_title("Node {}".format(node+1+nodes_plotted))
            axs[node].set_xlabel("Time(t)")
            axs[node].set_ylabel("Output(y)")
            axs[node].grid()
        plt.show()
        nodes_plotted += nodes_in_plot

