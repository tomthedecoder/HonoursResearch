import numpy as np
import matplotlib.pyplot as plt
from KuramotoOscillator import KuramotoOscillator
from CTRNN import CTRNN


def plot_all_nodes(network, final_t):
    """ Outputs a plot of each neuron"""

    if network.num_nodes <= 0:
        raise ValueError("Network must have at least one node")

    # reset ctrnn
    network.reset()

    times, y_output = network.evaluate(final_t)

    title = ''
    if isinstance(network, KuramotoOscillator):
        title = "Kuramoto Oscillator Node Output"
    elif isinstance(network, CTRNN):
        title = "CTRNN Node Output"

    if network.num_nodes == 1:
        plt.figure(figsize=(12, 12), dpi=100)
        plt.grid()
        plt.title(f'{title}\nNode 1', fontsize=20)
        plt.xlabel("Time(t)", fontsize=13)
        plt.ylabel("Output(y)", fontsize=13)
        plt.plot(times, y_output)
        return

    # maximum of 5 nodes to one plot
    MAX_NODE = 5
    nodes_plotted = 0
    for idx in range(int(np.ceil(network.num_nodes / MAX_NODE))):
        nodes_in_plot = min(network.num_nodes - nodes_plotted, MAX_NODE)
        fig, axs = plt.subplots(nodes_in_plot, figsize=(10, 10), dpi=100)
        fig.suptitle(title)

        if nodes_in_plot == 1:
            axs.plot(times, network.node_history[nodes_plotted])
            axs.set_title(f"Node {1 + nodes_plotted}", fontsize=13)
            axs.set_xlabel("Time(t)", fontsize=10)
            axs.set_ylabel("Output(y)", fontsize=10)
            axs.grid()
        else:
            for node in range(nodes_in_plot):
                axs[node].plot(times, network.node_history[node + nodes_plotted])
                axs[node].set_title(f"Node {node+1+nodes_plotted}", fontsize=13)
                axs[node].set_xlabel("Time(t)", fontsize=10)
                axs[node].set_ylabel("Output(y)", fontsize=10)
                axs[node].grid()
        nodes_plotted += nodes_in_plot

