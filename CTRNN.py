import numpy as np


def sigmoid(x):
    """ sigmoid activation function"""

    return 1.0 / (1.0 + np.exp(-x))


class Weight:
    def __init__(self, i, j, value):
        """ Container class to keep CTRNN.calculate_derivative O(n^2), consider the weight matrix doesn't need to be
            stored and later code can just iterate over self.genome, making a O(self.num_weights) calculation
            per evaluation"""

        self.i = i
        self.j = j
        self.value = value

    def get_i(self):
        return self.i

    def get_j(self):
        return self.j

    def __str__(self):
        return "({},{},{})".format(self.i, self.j, self.value)


class CTRNN(object):
    def __init__(self, num_nodes, genome, connection_array, num_weights=None, center_crossing=False):
        """ Implements a continuous time recurrent neural network"""

        self.num_nodes = num_nodes
        self.genome = genome

        # covers every case in order to initialse the number of weights and therefore the weight array
        if num_weights is None:
            self.num_weights = len(connection_array)
        else:
            self.num_weights = num_weights

        # initialase weights
        self.weights = []
        try:
            for idx, weight in enumerate(self.genome[0:self.num_weights]):
                i, j = connection_array[idx]
                self.weights.append(Weight(i, j, weight))
        except IndexError:
            print("Connection array's dimension must be equal to the number of weights")

        self.taus = self.genome[self.num_weights:self.num_weights + self.num_nodes]

        # if instructed to make center crossing weights, disregarded genome normally containing weights
        if center_crossing:
            self.biases = [0.0 for _ in range(self.num_nodes)]
            for weight in self.weights:
                bias_i = weight.get_i()
                j = weight.get_j()
                self.biases[j][bias_i] += weight.value / 2
        else:
            self.biases = self.genome[self.num_weights + self.num_nodes:]

        # there are self.num_nodes taus and biases
        self.num_genes = self.num_weights + 2 * self.num_nodes

        # array of node values
        self.node_values = np.array(0.0 * np.random.randn(self.num_nodes), dtype=np.float32)

        # array of derivatives w.r.t time for each node
        self.derivatives = np.array(np.zeros(self.num_nodes), dtype=np.float32)

        # the input to the nodes
        # in this implementation, input values to all nodes are the same
        self.forcing = 0

        # value of each node over time steps
        self.node_history = [[] for _ in range(self.num_nodes)]

        # determines values of nodes should be saved, as this can be quite expensive computationally
        self.save = False

        # step size for euler integration
        self.step_size = 0.01

    def __repr__(self):
        return "f'weights':\n{self.weights}\nbiases:\t{self.biases}\ntaus:\t{self.taus}"

    def reset(self, reset_history=False):
        """ Easy reset of CTRNN, setting node values, derivatives, last_time and forcing term are set to 0.
            Intended so that evaluation is non-biased"""

        self.node_values = np.array([0.0 for _ in range(self.num_nodes)])
        self.derivatives = np.array([0.0 for _ in range(self.num_nodes)])
        self.forcing = np.float(0.0)

        # only reset fully node_history if save is on
        if reset_history:
            self.node_history = [[] for _ in range(self.num_nodes)]

    def set_forcing(self, value):
        """ Sets the forcing term of node i to value"""

        self.forcing = value

    def node_value(self, node_i):
        """ Returns value of node_i"""

        return self.node_values[node_i]

    def calculate_derivative(self):
        """ Recalculates each derivative term"""

        sigmoid_terms = np.array([0.0 for _ in range(self.num_nodes)])

        # calculate the weight * sigmoid terms for each node
        for weight in self.weights:
            from_node = weight.get_i() - 1
            to_node = weight.get_j() - 1
            sigmoid_terms[from_node] += weight.value * sigmoid(self.node_values[to_node] + self.biases[to_node])

        self.derivatives = (-self.node_values + sigmoid_terms + self.forcing) / self.taus

        return self.derivatives

    def update(self):
        """ Updates each the value of each node"""

        self.calculate_derivative()
        self.node_values += self.derivatives * self.step_size

        # not enough memory sometimes and increases speed when record not needed
        if self.save:
            for index in range(self.num_nodes):
                self.node_history[index].append(self.node_values[index])


