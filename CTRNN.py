import numpy as np
from Individual import Individual
from CTRNNParameters import CTRNNParameters
from Network import Network


class CTRNN(Individual, Network):
    def __init__(self, ctrnn_parameters: CTRNNParameters):
        """ Implements a continuous time recurrent neural network"""

        Individual.__init__(self, ctrnn_parameters)
        Network.__init__(self, ctrnn_parameters)

    def y_prime(self, t, node_values):
        """ Recalculates the derivative of each term"""

        def sigmoid(x):
            """ Sigmoid activation function"""

            try:
                x = np.divide(1.0, (np.add(1.0, np.exp(-x))))
                return x
            except:
                return 0.0

        sigmoid_terms = np.array([0.0 for _ in range(self.params.num_nodes)])

        # calculate the weight * sigmoid terms for each node
        for weight in self.params.weights:
            i, j = weight.i, weight.j
            sigmoid_terms[i] += weight.value * sigmoid(node_values[j] + self.params.biases[j])

        self.set_forcing(t)
        self.derivatives = np.divide((-node_values + sigmoid_terms + self.forcing), self.params.taus)

        return self.derivatives

    def copy(self):
        copied_network = CTRNN(self.params.copy())
        copied_network.last_fitness = self.last_fitness

        return copied_network



