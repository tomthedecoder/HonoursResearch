from dataclasses import dataclass, field
import numpy as np


@dataclass(unsafe_hash=False)
class Genome:
    """ Holds the genome, i.e genetic material of an individual. Comes with getter and setter methods"""

    genome: field(init=True)

    def __post_init__(self):
        genome_type = type(self.genome)
        assert genome_type is list or genome_type is np.array
