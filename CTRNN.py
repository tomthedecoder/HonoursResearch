import numpy as np
from Individual import Individual
from CTRNNParameters import CTRNNParameters
from CTRNNStructure import CTRNNStructure

""""
Contributors to the CTRNN class:
Thomas Bailie
Mathew Egbert 
"""


def sigmoid(x):
    """ sigmoid activation function"""

    return 1.0 / (1.0 + np.exp(-x))


class CTRNN(Individual):
    def __init__(self, ctrnn_parameters):
        """ Implements a continuous time recurrent neural network"""

        assert type(ctrnn_parameters) is CTRNNParameters

        super(CTRNN, self).__init__(ctrnn_parameters)

        self.num_nodes = self.params.num_nodes
        self.num_weights = self.params.num_weights
        self.genome = self.params.genome

        self.node_values = np.array(np.zeros(self.params.num_nodes), dtype=np.float32)
        self.derivatives = np.array(np.zeros(self.params.num_nodes), dtype=np.float32)
        self.forcing = np.array([np.float(0.0) for _ in range(self.params.num_nodes)])
        self.node_history = [[] for _ in range(self.params.num_nodes)]

        self.step_size = np.float(0.05)
        self.last_time = np.float(0.0)

    def reset(self):
        """ Sets node values, derivatives, last_time and forcing term are set to 0."""

        self.node_values = np.array([np.float(0.0) for _ in range(self.params.num_nodes)])
        self.derivatives = np.array([np.float(0.0) for _ in range(self.params.num_nodes)])
        self.node_history = [[] for _ in range(self.params.num_nodes)]
        self.last_time = np.float(0.0)

    def set_forcing(self, t):
        """ Sets the forcing term of node i to value"""

        for i in range(self.params.num_nodes):
            self.forcing[i] = 0.0
            for j in range(self.params.num_forcing):
                self.forcing[i] += self.params.forcing_weights[i][j] * self.params.forcing_signals[i][j](t)

    def y_prime(self, t, node_values):
        """ Recalculates the derivative of each term"""

        sigmoid_terms = np.array([0.0 for _ in range(self.params.num_nodes)])

        # calculate the weight * sigmoid terms for each node
        for weight in self.params.weights:
            i, j = weight.i - 1, weight.j - 1
            sigmoid_terms[j] += weight.value * sigmoid(node_values[i] + self.params.biases[i])

        self.set_forcing(t)
        self.derivatives = np.divide((-node_values
                                      + sigmoid_terms
                                      + self.forcing),
                                      (self.params.taus + 0.001))

        return self.derivatives

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
        return solution.t, self.params.output_handler.call(self.node_history[-1])



