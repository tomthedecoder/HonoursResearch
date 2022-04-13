from scipy.integrate import solve_ivp
import numpy as np
from dataclasses import field


class Network:
    def __init__(self, parameters):
        """ Contains several generalisations which apply to both the kuramoto oscillator and the CTRNN"""

        self.params = parameters

        # easy to use aliases
        self.num_nodes = parameters.num_nodes
        self.num_weights = parameters.num_weights

        self.node_values = np.array(np.zeros(self.num_nodes), dtype=np.float32)
        self.derivatives = np.array(np.zeros(self.num_nodes), dtype=np.float32)
        self.forcing = np.array([np.float(0.0) for _ in range(self.num_nodes)])
        self.node_history = [[] for _ in range(self.num_nodes)]
        self.eval_times = []

        self.step_size = np.float(0.001)
        self.last_time = np.float(0.0)

    def reset(self):
        """ Sets node values, derivatives, last_time and forcing term are set to 0."""

        self.node_values = np.array([np.float(0.0) for _ in range(self.num_nodes)])
        self.derivatives = np.array([np.float(0.0) for _ in range(self.num_nodes)])
        self.node_history = [[] for _ in range(self.num_nodes)]
        self.params.eval_valid = False
        self.last_time = np.float(0.0)

    def set_forcing(self, t):
        """ Compute total input going to each neuron"""
        for i in range(self.num_nodes):
            self.forcing[i] = 0.0
            for j in range(self.params.num_forcing):
                self.forcing[i] += self.params.forcing_weights[i][j] * self.params.forcing_signals[i][j](t)

    def evaluate(self, final_t: float):
        """ Gets network output by RK45 method. Assumes that the last node is the output node. Evaluates in (0, final_t).
            t_space """

        if self.params.eval_valid:
            return self.eval_times, self.params.output_handler.call(self.node_history[-1])

        self.reset()

        if self.last_time > final_t:
            raise ValueError("Current time greater than final time")

        t_space = np.linspace(0, final_t, int(final_t / self.step_size))

        solution = solve_ivp(self.y_prime, t_span=(0, t_space[-1]), y0=self.node_values,
                             method='RK45', t_eval=t_space, dense_output=True, max_step=self.step_size)

        self.params.eval_valid = True
        self.node_history = solution.y
        self.eval_times = solution.t

        return self.eval_times, self.params.output_handler.call(self.node_history[-1])

    def y_prime(self, t, node_values):
        """ Governing equation of the network"""

        return

    def plot_parameters(self):
        """ Plots the network's"""

        return

    def copy(self):
        """Returns a copy of instance"""

        return Network(self.params.copy())


