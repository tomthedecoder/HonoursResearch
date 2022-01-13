import numpy as np
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
from os import path
from CTRNN import CTRNN


def fourier_ids(fourier_transform):
    """ Returns an array of fourier ids. It covers several cases in order to obtain the peek of the fourier transform"""

    ###########################################
    # Find the peeks of the fourier transform #
    ###########################################

    # find the set of points where a peek of the fourier transform is to the right of
    # a point is a boundary point if it is a minima or if it converges to a small enough radius
    # remove the last if statement if it causes program to stop searching fourier_transform prematurely
    peek_boundaries = []
    length = len(fourier_transform)
    for idx in range(1, length - 1):
        last_y = fourier_transform[idx - 1]
        next_y = fourier_transform[idx + 1]
        curr_y = fourier_transform[idx]
        if last_y >= curr_y and next_y >= curr_y:
            peek_boundaries.append(idx)
        if last_y >= curr_y >= next_y and abs(last_y - curr_y) < 0.001 and abs(curr_y - next_y) < 0.001:
            break

    # when fourier transforms are completely separated in frequency space the points, because of the additive property
    point_index = 0
    for idx, point in enumerate(fourier_transform):
        # a point is close to zero
        #print(idx, point)
        if point <= 0.003:
            # if the point close to zero is adjacent to last point close to zero, then a peek is between them
            #print('point', point, idx, point_index)
            if point_index+1 != idx and idx != 0:
                #print('indexes',point_index, idx)
                peek_boundaries.append(point_index)
                peek_boundaries.append(idx)
            point_index = idx

    ##############
    # Get the id #
    ##############

    # find the highest and second-highest value for the fourier transform within each boundary point
    fourier_ids = []
    boundary_length = len(peek_boundaries)
    for idb in range(boundary_length - 1):
        lhs_boundary = peek_boundaries[idb]
        rhs_boundary = peek_boundaries[idb + 1]
        second_highest = 0
        highest = 0
        for y_value in fourier_transform[lhs_boundary:rhs_boundary]:
            if highest < y_value:
                second_highest = highest
                highest = y_value
            elif second_highest < y_value:
                second_highest = y_value

        _id = highest - second_highest
        fourier_ids.append(_id)

    return fourier_ids


class LookUpTable:
    def __init__(self):
        """ Stores the fourier id's in a table"""

        self.table = {}

        if path.exists("lookup"):
            with open("lookup", "r") as reader:
                all_lines = reader.readlines()
                for idx in range(0, len(all_lines), 4):
                    fid = float(all_lines[idx])
                    num_nodes = int(all_lines[idx + 1])
                    genome = [float(num) for num in all_lines[idx + 2].split()]
                    connectivity_array = []
                    for entry in all_lines[idx + 3].split():
                        i, j = entry.split(',')
                        connection = (int(i), int(j))
                        connectivity_array.append(connection)

                    ctrnn = CTRNN(num_nodes, genome,  connectivity_array)
                    self.table[fid] = ctrnn

    def query(self, fourier_id):
        """ Maps id to ctrnn"""

        if fourier_id in self.table:
            return self.table[fourier_id]
        else:
            return False

    def add(self, fid, ctrnn):
        """ Adds instance of ctrnn to table"""

        self.table[fid] = ctrnn

    def save(self):
        """ Adds fourier id to the lookup table, the format is
            id, num_nodes, genome, connectivity_array per line"""

        table = ""
        for fid in self.table:
            ctrnn = self.table[fid]

            # save genome
            genome = ""
            for gene in ctrnn.genome:
                genome += f"{gene} "

            # for connection array
            connection_array = ""
            for connection in ctrnn.connection_array:
                from_node, to_node = connection
                connection_array += f"{from_node},{to_node} "

            table += f"{fid}\n{ctrnn.num_nodes}\n{genome}\n{connection_array}\n"

        with open("lookup", "w") as writer:
            writer.write(table)

    def __iter__(self):
        return iter(self.table)


test_table = False
if test_table:
    table = LookUpTable()

    # add ctrnn to table
    num_nodes = 1
    genome = [1, 1, 1]
    connectivity_array = [(0, 0)]

    ctrnn = CTRNN(num_nodes, genome, connectivity_array)
    fid = 1

    table.add(fid, ctrnn)
    table.save()

make_file = True
if make_file:
    table = LookUpTable()

    def target_signal(t):
        return np.sin(t) + np.sin(2*t)

    # Number of sample points
    N = 10000

    # sample spacing
    T = 5 * np.floor(np.pi) / 2*N
    t = np.linspace(0.0, N * T, N, endpoint=False)
    y = target_signal(t)
    yf = fft(y)
    xf = fftfreq(N, T)[:N // 2]
    plot_points = 2.0 / N * np.abs(yf[0:N // 2])

    plt.figure()
    plt.subplot(211)
    plt.plot(xf, plot_points, color='tab:blue', marker='o')
    plt.plot(xf, plot_points, color='black')

    plt.subplot(212)
    plt.plot(t, y, color='tab:orange')

    plt.show()

    # add peek ids to table, but no ctrnns!

    for fid in fourier_ids(plot_points):
        table.add(fid, "instance")