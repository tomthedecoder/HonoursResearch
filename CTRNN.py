import numpy as np
from Individual import Individual

""""
Contributors to the CTRNN class:
Thomas Bailie
Mathew Egbert 
"""


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

    def traveled_from(self):
        return self.i

    def traveled_to(self):
        return self.j

    def __str__(self):
        return "({},{},{})".format(self.i, self.j, self.value)


class CTRNN(Individual):
    def __init__(self, num_nodes, genome, connection_array):
        """ Implements a continuous time recurrent neural network"""

        super(CTRNN, self).__init__(genome, num_nodes)

        self.num_nodes = num_nodes
        self.genome = genome
        self.connection_array = connection_array
        self.num_weights = len(connection_array)

        # initialize weights
        self.weights = []
        for idx, weight in enumerate(self.genome[0:self.num_weights]):
            i, j = self.connection_array[idx]
            self.weights.append(Weight(i, j, weight))

        self.taus = np.array(self.genome[self.num_weights:self.num_weights + self.num_nodes])
        self.biases = np.array(self.genome[self.num_weights + self.num_nodes:self.num_weights + 2 * self.num_nodes])
        self.forcing_weights = np.array(self.genome[self.num_weights + 2 * self.num_nodes:self.num_weights])
        #self.gains = np.array

        # there are self.num_nodes taus and biases and input weights
        self.num_genes = self.num_weights + 3 * self.num_nodes

        # array of node values
        self.node_values = np.array(np.zeros(self.num_nodes), dtype=np.float32)

        # array of derivatives w.r.t time for each node
        self.derivatives = np.array(np.zeros(self.num_nodes), dtype=np.float32)

        # the input to the nodes
        # in this implementation, input values to all nodes are the same
        self.forcing = [0.0 for _ in range(self.num_nodes)]

        # value of each node over time steps
        self.node_history = [[] for _ in range(self.num_nodes)]

        # step size for euler integration
        self.step_size = 0.01

        # times of evaluation
        self.last_time = 0

    def __repr__(self):
        return "f'weights':\n{self.weights}\nbiases:\t{self.biases}\ntaus:\t{self.taus}"

    def reset(self):
        """ Sets node values, derivatives, last_time and forcing term are set to 0."""

        self.node_values = np.array([0.0 for _ in range(self.num_nodes)])
        self.derivatives = np.array([0.0 for _ in range(self.num_nodes)])
        self.node_history = [[] for _ in range(self.num_nodes)]
        self.forcing = [0.0 for _ in range(self.num_nodes)]
        self.last_time = 0

    def set_forcing(self, t):
        """ Sets the forcing term of node i to value"""

        for idx in range(self.num_nodes):
            self.forcing[idx] = 0
            for idy in range(self.num_inputs):
                self.forcing[idx] += self.forcing_weights[idy] * self.input_signals[idy](t)

    def get_forcing(self, node_i):
        """ Returns the forcing term for node_i"""

        return self.forcing[node_i]

    def node_value(self, node_i):
        """ Returns value of node_i"""

        return self.node_values[node_i]

    def calculate_derivative(self):
        """ Recalculates the derivative of each term"""

        sigmoid_terms = np.array([0.0 for _ in range(self.num_nodes)])
        # calculate the weight * sigmoid terms for each node
        for weight in self.weights:
            from_node = weight.traveled_from() - 1
            to_node = weight.traveled_to() - 1
            sigmoid_terms[to_node] += weight.value * sigmoid(self.node_values[from_node] + self.biases[from_node])

        self.derivatives = (-self.node_values + sigmoid_terms + self.forcing_weights * self.forcing) / self.taus
        #print(self.forcing_weights)
        return self.derivatives

    def update(self):
        """ Updates the value of each node"""

        self.calculate_derivative()
        #print(self.derivatives, self.node_values)
        self.node_values += self.derivatives * self.step_size

        for index in range(self.num_nodes):
            self.node_history[index].append(self.node_values[index])

    def set_parameter(self, pos, new_value):
        """ Sets the parameter at pos to the new parameter"""

        self.genome[pos] = new_value
        if pos <= self.num_weights - 1:
            i_existing = self.weights[pos].i
            j_existing = self.weights[pos].j
            self.weights[pos] = Weight(i_existing, j_existing, new_value)
        elif self.num_weights - 1 < pos <= self.num_weights + self.num_nodes - 1:
            self.taus[pos - self.num_weights] = new_value
        elif self.num_weights + self.num_nodes - 1 <= pos < self.num_weights + 2 * self.num_nodes - 1:
            self.biases[pos - self.num_weights - self.num_nodes] = new_value
        elif self.num_weights + 2 * self.num_nodes - 1 <= pos:
            self.forcing_weights[pos - self.num_weights - 2 * self.num_nodes] = new_value
        else:
            raise IndexError("Index exceeds number of parameter in CTRNN")

    def evaluate(self, final_t):
        """ Gets CTRNN output by euler-step method. Assumes that the last node is the output node"""

        self.reset()

        if self.last_time > final_t:
            raise ValueError("Current time greater than final time")

        num_steps = final_t / self.step_size
        for _ in range(int(num_steps)):
            self.set_forcing(self.last_time)
            self.update()
            self.last_time += self.step_size

        return self.node_history[-1]



