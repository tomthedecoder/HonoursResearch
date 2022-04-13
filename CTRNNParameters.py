from dataclasses import dataclass
from OutputHandler import OutputHandler
from Weight import Weight
from Parameters import *
import numpy as np


class CTRNNParameters(Parameters):
    def __init__(self, genome, output_handler, forcing_signals, connection_array):
        """ Parameters for the CTRNN class"""

        super(CTRNNParameters, self).__init__(genome, output_handler, forcing_signals, len(connection_array))

        self.weights = []
        for idx, weight in enumerate(self.genome[0:self.num_weights]):
            i, j = connection_array[idx]
            self.weights.append(Weight(i, j, weight))

        self.taus = np.array(self.genome[self.num_weights:self.num_weights + self.num_nodes])
        self.biases = np.array(self.genome[self.num_weights + self.num_nodes:self.num_weights + 2 * self.num_nodes])
        self.forcing_weights = np.array(
                         [self.genome[self.num_weights + 2 * self.num_nodes + i * self.num_forcing:
                          self.num_weights + 2 * self.num_nodes + (i + 1) * self.num_forcing]
                          for i in range(self.num_nodes)])

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
        elif pos < self.num_weights + self.num_nodes * (2 + self.num_forcing):
            p = pos - self.num_weights - 2 * self.num_nodes
            self.forcing_weights[p // self.num_forcing][p // self.num_nodes] = new_value
        else:
            raise IndexError("Index exceeds number of parameter in CTRNN")

        self.eval_valid = False

    def __str__(self):
        weights = [(weight.i, weight.j, weight.value) for weight in self.weights]
        return f"weights: {weights}\ntaus: {self.taus}\nbiases: {self.biases}\nforcing weights: {self.forcing_weights}"

    def copy(self):
        """ Returns copy of instance"""

        connection_array = []
        for weight in self.weights:
            connection_array.append((weight.i, weight.j))

        genome = deep_copy(self.genome)

        return CTRNNParameters(genome, self.output_handler, self.forcing_signals, connection_array)