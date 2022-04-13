from NetworkStructure import *
from Distribution import Distribution
from OutputHandler import OutputHandler
import numpy as np


@dataclass(unsafe_hash=True)
class CTRNNStructure(NetworkStructure):
    """ Given to an environment to make a CTRNNs with"""

    center_crossing: bool = field(default=False)
