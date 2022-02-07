import numpy as np
from CTRNNStructure import *
from Environment import *
from plot_all_neurons import plot_all_neurons
from genome_distribution import *
from sys import stdout
from time import time
import matplotlib.pyplot as plt
from OutputHandler import *

cross_over_probability = 0.5
num_demes = 4
num_nodes = 2
pop_size = 2
mutation_chance = 0.9
connectivity_array = None
demes = []

final_t = np.ceil(3 * np.pi)
fitness_type = "simpsons"
cross_over_type = "microbial"
distribution_type = "uniform"
handler_type = "default"
connection_type = "default"
num_generations = 1
num_runs = 1

signals = [lambda t: np.sin(t)]

mask_type = ""
forcing_signals = make_signals(num_nodes, signals)
if mask_type == "not last":
    forcing_signals = apply_forcing_mask(num_nodes, forcing_signals, not_last_mask(num_nodes, len(signals)))
elif mask_type == "first node":
    forcing_signals = apply_forcing_mask(num_nodes, forcing_signals, first_node_mask(num_nodes, len(signals)))

lows, highs = uniform_parameters()
lambdas = poisson_parameters()
mus, stds = normal_parameters()