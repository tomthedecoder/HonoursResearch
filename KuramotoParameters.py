from Parameters import *
import numpy as np


class KuramotoParameters(Parameters):
    """ Contains the parameters of a kuramoto oscillator"""

    def __init__(self, genome, output_handler, forcing_signals, num_weights):
        super(KuramotoParameters, self).__init__(genome, output_handler, forcing_signals, num_weights)

        self.natural_frequencies = self.genome[0:self.num_nodes]
        self.k = self.genome[self.num_nodes:].reshape((self.num_nodes, self.num_nodes))

    def set_parameter(self, pos, new_value):
        self.genome[pos] = new_value
        if pos < self.num_nodes:
            self.natural_frequencies[pos] = new_value
        elif pos < self.num_nodes + self.num_nodes ** 2:
            pos = pos - self.num_nodes
            self.k[pos // self.num_nodes][pos // self.num_nodes] = new_value
        else:
            raise ValueError("Index exceeds number of parameter in Kuramoto oscillator")

        self.eval_valid = False

    def __str__(self):
        return f"natural frequencies: {self.natural_frequencies}\nk:{self.k}"

    def copy(self):
        """ Returns a copied Kuramoto Parameters from instance"""

        natural_frequencies = deep_copy(self.natural_frequencies)
        k = deep_copy(self.k)

        genome = np.append(natural_frequencies, k)

        return KuramotoParameters(genome, self.output_handler, self.forcing_signals, self.num_weights)