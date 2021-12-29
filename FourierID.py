from scipy.fft import fft, fftfreq
import numpy as np
import matplotlib.pyplot as plt
import os


class FourierID:
    def __init__(self, _id):
        """ An ID given to a peek in a fourier transformation"""

        self.id = _id

    @staticmethod
    def fourier_ids(fourier_transform):
        """ Returns an array of fourier ids"""

        # find the set of points where a peek of the fourier transform is to the right of
        # a point is a boundary point if it is a minima or if it converges to a small enough radius
        peek_boundaries = [0]
        length = len(fourier_transform)
        for idx in range(1, length-1):
            last_y = fourier_transform[idx - 1]
            next_y = fourier_transform[idx + 1]
            curr_y = fourier_transform[idx]
            if curr_y <= next_y and curr_y <= last_y:
                peek_boundaries.append(idx)
            if abs(next_y - curr_y) < 0.001:
                peek_boundaries.append(idx)
                break

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
            fourier_ids.append(FourierID(_id))

        return fourier_ids

    def get_id(self):
        return self.id

    def set_id(self, new_id):
        self.id = new_id


make_file = True
if make_file:
    from scipy.stats import norm

    def target_signal(t):
        return np.sin(np.pi * t) + np.sin(5 * np.pi * t) #+ np.sin(t + 100)#+ norm.rvs(loc=0, scale=0.05, size=len(t))

    # Number of sample points
    N = 1000
    # sample spacing
    T = 5 * np.floor(np.pi) / N
    t = np.linspace(0.0, N * T, N, endpoint=False)
    y = target_signal(t)
    yf = fft(y)
    xf = fftfreq(N, T)[:N // 2]
    plot_points = 2.0 / N * np.abs(yf[0:N // 2])
    fourier_ids = FourierID.fourier_ids(plot_points)
    for fid in fourier_ids:
        print(fid.id)
    contents = ""
    for idx, p in enumerate(plot_points):
        contents += "{},{}\n".format(xf[idx], p)

    with open("fourier", "w") as write_file:
        write_file.write(contents)

    plt.figure()
    plt.subplot(211)
    plt.plot(xf, plot_points, color='tab:blue', marker='o')
    plt.plot(xf, plot_points, color='black')

    plt.subplot(212)
    plt.plot(t, target_signal(t), color='tab:orange')

    plt.show()