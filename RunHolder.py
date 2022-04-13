from DemeContainer import *
from Deme import *
import functools
from dataclasses import field, dataclass


@dataclass(unsafe_hash=True)
class RunHolder:
    num_demes: int
    num_runs: int
    num_networks: int
    cross_over_type: str
    fitness_type: str

    def __post_init__(self):
        self.demes = [[RingTopology(self.cross_over_type, self.fitness_type, self.num_demes) for _ in range(self.num_runs)] for _ in range(self.num_networks)]

        self.best_demes = []
        self.best_networks = []

        self.best_deme_ids = [0 for _ in range(self.num_networks)]
        self.best_runs = [0 for _ in range(self.num_networks)]

        self.best_fitness = [[[] for _ in range(self.num_runs)] for _ in range(self.num_networks)]
        self.average_fitness = [[[[] for _ in range(self.num_demes)] for _ in range(self.num_runs)] for _ in range(self.num_networks)]

    def add(self, net_i: int, run_i: int, deme: Deme):
        self.demes[net_i][run_i].append(deme)

    def set(self, net_i: int, run_i: int, deme_i: int, deme: Deme):
        self.demes[net_i][run_i][deme_i] = deme

    def get(self, net_i, run_i, deme_i):
        return self.demes[net_i][run_i][deme_i]

    def get_topology(self, net_i: int):
        return self.demes[net_i]

    def get_demes(self, net_i: int, run_i: int):
        return self.demes[net_i][run_i]

    def init_best(self):
        """ Initializes the best networks"""

        self.best_demes = [self.demes[i][0][0] for i in range(self.num_networks)]
        self.best_networks = [self.best_demes[i].individuals[-1] for i in range(self.num_networks)]