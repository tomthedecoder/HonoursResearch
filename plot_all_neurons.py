import numpy as np
import matplotlib.pyplot as plt


def plot_all_neurons(ctrnn, final_t=10):
    """ Outputs a plot of each neuron"""

    if ctrnn.num_nodes <= 0:
        raise ValueError("CTRNN must have at least one node")

    # reset ctrnn
    ctrnn.reset()

    times, y_output = ctrnn.evaluate(final_t=final_t)

    if ctrnn.num_nodes == 1:
        plt.figure()
        plt.title("Node 1")
        plt.grid()
        plt.xlabel("Time(t)")
        plt.ylabel("Output(y)")
        plt.plot(times, y_output)
        return

    # maximum of 5 nodes to one plot
    MAX_NODE = 5
    nodes_plotted = 0
    for idx in range(int(np.ceil(ctrnn.num_nodes / MAX_NODE))):
        nodes_in_plot = min(ctrnn.num_nodes - nodes_plotted, MAX_NODE)
        fig, axs = plt.subplots(nodes_in_plot)

        if nodes_in_plot == 1:
            axs.plot(times, ctrnn.node_history[nodes_plotted])
            axs.set_title(f"Node {1 + nodes_plotted}")
            axs.set_xlabel("Time(t)")
            axs.set_ylabel("Output(y)")
            axs.grid()
        else:
            for node in range(nodes_in_plot):
                axs[node].plot(times, ctrnn.node_history[node + nodes_plotted])
                axs[node].set_title(f"Node {node+1+nodes_plotted}")
                axs[node].set_xlabel("Time(t)")
                axs[node].set_ylabel("Output(y)")
                axs[node].grid()
        nodes_plotted += nodes_in_plot

