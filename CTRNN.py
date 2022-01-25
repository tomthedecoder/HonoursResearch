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
        self.forcing_weights = np.array(self.genome[self.num_weights + 2 * self.num_nodes:self.num_weights + 3 * self.num_nodes])
        self.shift = genome[-1]

        # there are self.num_nodes taus and biases and input weights
        self.num_genes = self.num_weights + 3 * self.num_nodes + 1

        # array of node values
        self.node_values = np.array(np.zeros(self.num_nodes), dtype=np.float32)

        # array of derivatives w.r.t time for each node
        self.derivatives = np.array(np.zeros(self.num_nodes), dtype=np.float32)

        # the input to the nodes
        # in this implementation, input values to all nodes are the same
        self.forcing = [np.float(0.0) for _ in range(self.num_nodes)]

        # value of each node over time steps
        self.node_history = [[] for _ in range(self.num_nodes)]

        # step size for euler integration
        self.step_size = np.float(0.01)

        # times of evaluation
        self.last_time = np.float(0.0)

    def reset(self):
        """ Sets node values, derivatives, last_time and forcing term are set to 0."""

        self.node_values = np.array([np.float(0.0) for _ in range(self.num_nodes)])
        self.derivatives = np.array([np.float(0.0) for _ in range(self.num_nodes)])
        self.node_history = [[] for _ in range(self.num_nodes)]
        self.last_time = np.float(0.0)

    def set_forcing(self, t):
        """ Sets the forcing term of node i to value"""

        for idx in range(self.num_nodes):
            self.forcing[idx] = self.input_signals[0](t)

    def get_forcing(self, node_i):
        """ Returns the forcing term for node_i"""

        return self.forcing[node_i]

    def node_value(self, node_i):
        """ Returns value of node_i"""

        return self.node_values[node_i]

    def y_prime(self, t, node_values):
        """ Recalculates the derivative of each term"""

        sigmoid_terms = np.array([0.0 for _ in range(self.num_nodes)])

        # calculate the weight * sigmoid terms for each node
        for weight in self.weights:
            i, j = weight.i - 1, weight.j - 1
            # because of transformation
            if i == self.num_genes - 1:
                bias = self.biases[i] - self.shift
            else:
                bias = self.biases[i]

            sigmoid_terms[j] += weight.value * sigmoid(node_values[i] + bias)

        self.set_forcing(t)
        self.derivatives = (-node_values + sigmoid_terms + self.forcing * self.forcing_weights) / (self.taus + 0.01)
        self.derivatives[-1] -= self.shift / (self.taus[-1] + 0.01)

        return self.derivatives

    def set_parameter(self, pos, new_value):
        """ Sets the parameter at pos to the new parameter"""

        self.genome[pos] = new_value
        if pos < self.num_weights:
            i_existing = self.weights[pos].i
            j_existing = self.weights[pos].j
            self.weights[pos] = Weight(i_existing, j_existing, new_value)
        elif pos < self.num_weights + self.num_nodes:
            self.taus[pos - self.num_weights] = new_value
        elif pos < self.num_weights + 2 * self.num_nodes:
            self.biases[pos - self.num_weights - self.num_nodes] = new_value
        elif pos < self.num_weights + 3 * self.num_nodes:
            self.forcing_weights[pos - self.num_weights - 2 * self.num_nodes] = new_value
        elif pos == self.num_genes - 1:
            self.shift = new_value
        else:
            raise IndexError("Index exceeds number of parameter in CTRNN")

    def evaluate(self, final_t):
        """ Gets CTRNN output by euler-step method. Assumes that the last node is the output node"""

        from scipy.integrate import solve_ivp

        self.reset()

        if self.last_time > final_t:
            raise ValueError("Current time greater than final time")

        t_space = np.linspace(0, final_t, int(final_t/self.step_size))
        solution = solve_ivp(self.y_prime, t_span=(0, t_space[-1]), y0=self.node_values,
                             method='RK45', t_eval=t_space,
                             dense_output=True, max_step=self.step_size)
        self.node_history = solution.y
        return solution.t, self.node_history[-1]



