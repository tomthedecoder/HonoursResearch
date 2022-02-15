from dataclasses import dataclass, field
from Distribution import Distribution
from OutputHandler import OutputHandler
import numpy as np


def general_make_topology(num_nodes, rule):
    """ General format for the creation of any toplogy"""

    connection_array = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if not rule(i, j):
                continue
            tup = (i, j)
            connection_array.append(tup)

    return connection_array


def weak_line_topology(num_nodes):
    """ Makes a connection array with the line topology. No self connections"""

    return general_make_topology(num_nodes, lambda i, j: j == i + 1)


def strong_line_topology(num_nodes):
    """Makes a connection array with the line topology except with self connections"""

    return general_make_topology(num_nodes, lambda i, j: j == i + 1 and j == i)


def fully_connected_topology(num_nodes):
    """ All nodes share a connection, including with it's self"""

    return general_make_topology(num_nodes, lambda i, j: True)


def probabilistically_delete_off_diagonals(num_nodes, p):
    """ Easy way to make connection array, p1 is the probability of deleting an edge between i and j where i != j"""

    return general_make_topology(num_nodes, lambda i, j: i != j and np.random.randint(0, 2) > p)


def apply_forcing_mask(num_nodes, forcing_signals, mask):
    """ Forcing signals is an array that contains the forcing signals for the CTRNN"""

    assert len(forcing_signals) == len(mask) == num_nodes > 0

    for i, row in enumerate(mask):
        for j, value in enumerate(row):
            forcing_signals[i][j] = forcing_signals[i][j] if value == 1.0 else lambda t: 0.0

    return forcing_signals


def make_signals(num_nodes, forcing_signals):
    """ Returns an array/vector of forcing signals from possible ones, e.g
        make_signals(3, cos(t)) will return [cos(t), cos(t), cos(t)]"""

    return np.array([[fs for fs in forcing_signals] for _ in range(num_nodes)])


def general_make_mask(num_nodes, num_inputs, rule):
    mask = []
    for i in range(num_nodes):
        mask.append([])
        for _ in range(num_inputs):
            if rule(i, num_nodes):
                d = 1.0
            else:
                d = 0.0

            mask[-1].append(d)

    return np.array(mask)


def first_node_mask(num_nodes, num_inputs):
    """ Generates an input mask such that only the first node has any input"""

    return general_make_mask(num_nodes, num_inputs, lambda i, num_nodes: i == 0)


def n_node_mask(num_nodes, num_inputs, n):
    """ input only on the first n nodes"""

    assert n <= num_nodes

    return general_make_mask(num_nodes, num_inputs, lambda i, num_nodes: i <= n - 1)


def not_last_mask(num_nodes, num_inputs):
    """ Generates mask where the input of the last node is set to 0"""

    return general_make_mask(num_nodes, num_inputs, lambda i, num_nodes: i < num_nodes - 1)


def non_mask(num_nodes, num_inputs):
    """No inputs"""

    return general_make_mask(num_nodes, num_inputs, lambda i, j: False)


@dataclass(unsafe_hash=True)
class CTRNNStructure:
    """ Given to an environment to make a CTRNN with"""

    distribution: Distribution
    num_nodes: int
    connection_array: list = field(default_factory=list)
    connection_type: str = field(default="default")
    center_crossing: bool = field(default=bool)

    def __post_init__(self):
        # default topology is a fully connected circuit
        if len(self.connection_array) == 0:
            if self.connection_type == "default":
                self.connection_array = fully_connected_topology(self.num_nodes)
            elif self.connection_type == "weak line":
                self.connection_array = weak_line_topology(self.num_nodes)
            elif self.connection_type == "strong line":
                self.connection_array = strong_line_topology(self.num_nodes)
            elif self.connection_type == "prob_delete":
                self.connection_array = probabilistically_delete_off_diagonals(self.num_nodes, p=0.25)
            else:
                raise ValueError("connection type not recognised with connection array of length 0")