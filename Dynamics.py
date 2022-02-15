import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from plot_all_neurons import *
from CTRNN import CTRNN


class Dynamics:
    def __init__(self, W, T, B, IW):
        self.weights = W
        self.taus = T
        self.biases = B
        self.forcing_weights = IW

