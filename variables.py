from CTRNNStructure import *
from Environment import *
from OutputHandler import *
from genome_distribution import *
from plot_all_neurons import plot_all_neurons
from sys import stdout
from time import time
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings("error")
np.seterr(over='warn')

connectivity_array = None
demes = []

cross_over_probability = 0.6
num_demes = 3
num_nodes = 4
pop_size = 200
mutation_chance = 0.9
num_generations = 500
num_runs = 4
final_t = np.ceil(5 * np.pi)

fitness_type = "simpsons"
cross_over_type = "microbial"
distribution_type = "uniform"
handler_type = "max/min"
connection_type = "default"
mask_type = "n node"

signals = [lambda t: 0]

forcing_signals = make_signals(num_nodes, signals)
if mask_type == "not last":
    forcing_signals = apply_forcing_mask(num_nodes, forcing_signals, not_last_mask(num_nodes, len(signals)))
elif mask_type == "n node":
    forcing_signals = apply_forcing_mask(num_nodes, forcing_signals, n_node_mask(num_nodes, len(signals), 3))
elif mask_type != "default":
    raise ValueError("Invalid mask type")

lows, highs = uniform_parameters()
lambdas = poisson_parameters()
mus, stds = normal_parameters()