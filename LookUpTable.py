import os
import numpy as np
from FourierID import FourierID


class LookUpTable:
    def __init__(self):
        """ Stores the fourier id's in a table"""

        self.table = {}

        with open("lookup", "r") as reader:
            for line in reader.readlines():
                if line == "\n":
                    continue
                line = line.split(',')
                _id = float(line[0])
                ctrnn = line[1]
                self.table[_id] = [FourierID(_id), ctrnn]

    def query(self, fourier_id):
        """ Maps fourier_id to function approximator, i.e the CTRNN by returning path to it's location"""

        if fourier_id in self.table:
            return self.table[fourier_id]
        else:
            return False

    def add(self, fourier_ids):
        """ Adds instance(s) of FourierID in fourier_ids to self.table"""

        if type(fourier_ids) is not list or type(fourier_ids) is not np.array:
            fourier_ids = [fourier_ids]

        for fid in fourier_ids:
            self.table[fid.id] = [fid, "instance"]

    def save(self):
        """ Adds fourier id to the lookup table"""

        table = ""
        if os.path.exists("lookup"):
            reader = open("lookup", "r")
            table = reader.read()

        for fourier_id in self.table:
            _id, location = self.table[fourier_id]
            table += str(_id) + str(location) + "\n"

        with open("lookup", "w") as writer:
            writer.write(table)