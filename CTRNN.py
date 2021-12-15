from scipy.integrate import solve_ivp
import numpy as np


def sigmoid(x):
    """ sigmoid activation function"""

    return 1.0 / (1.0 + np.exp(-x))


class CTRNN(object):
    def __init__(self, num_nodes, num_weights, genome):
        """ Implements a continuous time recurrent neural network"""

        self.num_nodes = num_nodes
        self.genome = genome
        self.num_weights = num_weights

        self.weights = self.genome[0:self.num_weights]
        self.taus = self.genome[self.num_weights:self.num_weights + self.num_nodes]
        self.biases = self.genome[self.num_weights + self.num_nodes:]

        # there are self.num_nodes taus and biases
        self.num_genes = self.num_weights + 2 * self.num_nodes

        # array of node values
        self.node_values = np.array(0.0 * np.random.randn(self.num_nodes), dtype=np.float32)

        # array of derivatives w.r.t time for each node
        self.derivatives = np.array(np.zeros(self.num_nodes), dtype=np.float32)

        # input values to some node
        self.inputs = np.zeros(self.num_nodes)

        # value of each node over time steps
        self.history = [[] for _ in range(self.num_nodes)]

        # determines values of nodes should be saved, as this can be quite expensive computationally
        self.save = False

        # step size for euler integration
        self.step_size = 0.01

    def __repr__(self):
        return f'weights:\n{self.weights}\nbiases:\t{self.biases}\ntaus:\t{self.taus}'

    def make_biases_centre_crossing(self):
        """ Changes biases to center crossing according to value currently stored in self.inputs"""

    def set_input(self, node_i, value):
        """ Sets the forcing term of node i to value"""

        self.inputs[node_i] = value

    def node_value(self, node_i):
        """ Returns value of node_i"""

        return self.node_values[node_i]

    def calculate_derivative(self):
        """ Recalculates each derivative term"""
        synaptic_inputs = [self.weights * sigmoid(self.node_values + self.biases)]
        self.derivatives = (-self.node_values + np.sum(synaptic_inputs, axis=1) + self.inputs) / self.taus

        return self.derivatives

    def update(self):
        """ updates each the value of each node"""

        self.calculate_derivative()
        self.node_values += self.derivatives * self.step_size

        # not enough memory sometimes
        if self.save:
            for index in range(self.num_nodes):
                self.history[index].append(self.node_values[index])


test = True
if test:
    num_weights = 1
    num_nodes = 1
    genome = [1, 1, 1]

    ctrnn = CTRNN(num_nodes, num_weights, genome)
    ctrnn.calculate_derivative()
    ctrnn.update()


