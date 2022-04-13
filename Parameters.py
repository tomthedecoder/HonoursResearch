
def deep_copy(array):
    """ Deep copies an array of primitive types"""

    return [x for x in array]


class Parameters:
    def __init__(self, genome, output_handler, forcing_signals, num_weights):
        """ Virtual class for the variables of a network and individual"""

        self.genome = genome
        self.output_handler = output_handler
        self.forcing_signals = forcing_signals

        self.num_nodes = len(self.forcing_signals)
        self.num_forcing = len(self.forcing_signals[0])
        self.num_genes = len(self.genome)
        self.num_weights = num_weights

        self.forcing_weights = [[0.0 for _ in range(self.num_forcing)] for _ in range(self.num_nodes)]
        self.eval_valid = False

    def set_parameter(self, pos, new_value):
        pass

    def copy(self):
        pass
