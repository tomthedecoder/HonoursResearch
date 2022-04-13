import numpy as np
from KuramotoParameters import KuramotoParameters
from Individual import Individual
from Network import Network


class KuramotoOscillator(Individual, Network):
    def __init__(self, kuramoto_parameters: KuramotoParameters, connection_array):
        Individual.__init__(self, kuramoto_parameters)
        Network.__init__(self, kuramoto_parameters)

        self.connection_array = connection_array

    def y_prime(self, t: float, theta: np.array):
        self.set_forcing(t)
        synaptic_terms = [0.0 for _ in range(self.params.num_nodes)]

        for tup in self.connection_array:
            i, j = tup
            synaptic_terms[i] += self.params.k[j][i] * np.sin(theta[j] - theta[i])

        self.derivatives = np.add(self.params.natural_frequencies, np.multiply(1/self.params.num_nodes, synaptic_terms))

        return self.derivatives

    def copy(self):
        copied_oscillator = KuramotoOscillator(self.params.copy(), self.connection_array)
        copied_oscillator.last_fitness = self.last_fitness

        return copied_oscillator

    def evaluate(self, final_t: float):
        """ Wrap around y-coordinate of circle"""

        times, output = super().evaluate(final_t)
        for i, nhistory in enumerate(self.node_history):
            self.node_history[i] = np.sin(nhistory)

        return times, self.node_history[-1]