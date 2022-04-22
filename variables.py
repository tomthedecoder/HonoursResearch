from CTRNNStructure import *
from Deme import *
from OutputHandler import *
from KuramotoParameters import *
from KuramotoOscillator import *
from KuramotoStructure import *
from Network import *
from genome_distribution import *
from DemeContainer import *
from RunHolder import *
from TargetSignal import *
from RunHolder import *
from plot_all_neurons import plot_all_nodes
from sys import stdout
from time import time
import matplotlib.pyplot as plt
from DemeContainer import *
import os
import numpy as np
import warnings
import multiprocessing as mp

warnings.filterwarnings("error")
np.seterr(over='warn')

connectivity_array = None

cross_over_probability = 0.6
num_demes = 1
num_nodes = 2
pop_size = 5
num_generations = 5
num_runs = 1
final_t = 2.65
start_t = 0.8

fitness_type = "1/simpsons"
cross_over_type = "microbial"
distribution_type = "uniform"
network_types = ["kuramoto"]
handler_type = "max/min"
connection_type = "default"
mask_type = "no forcing"

num_networks = len(network_types)

signals = [lambda t: np.sin(t)]

forcing_signals = make_signals(num_nodes, signals)
if mask_type == "not last":
    forcing_signals = apply_forcing_mask(num_nodes, forcing_signals, not_last_mask(num_nodes, len(signals)))
elif mask_type == "n node":
    forcing_signals = apply_forcing_mask(num_nodes, forcing_signals, n_node_mask(num_nodes, len(signals), 3))
elif mask_type == "no forcing":
    forcing_signals = apply_forcing_mask(num_nodes, forcing_signals, no_forcing(num_nodes, len(signals)))
elif mask_type != "default":
    raise ValueError("Invalid mask type")

lows, highs = ctrnn_uniform_parameters()
lambdas = poisson_parameters()
mus, stds = normal_parameters()