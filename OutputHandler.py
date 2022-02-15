from sklearn.preprocessing import minmax_scale, scale, maxabs_scale
import numpy as np


def mean_shift(y_values):
    """ Scaling method which sends x to x - mu"""

    start_point = int(10 * len(y_values) / 100)
    mu = np.mean(y_values[start_point:])

    return np.array([y - mu for y in y_values])


class OutputHandler:
    def __init__(self, method='max/min'):
        """ Scales the output of the CTRNN's last node"""

        self._method = method.strip()

    def call(self, y_values):
        if self._method == 'max/min':
            return minmax_scale(y_values, feature_range=(-1, 1))
        elif self._method == 'white':
            return scale(y_values)
        elif self._method == 'maxabs':
            return maxabs_scale(y_values)
        elif self._method == 'mean shift':
            return mean_shift(y_values)
        elif self._method == 'max/min&mean shift':
            return mean_shift(minmax_scale(y_values, feature_range=(-1, 1)))
        elif self._method == 'default':
            return y_values
        else:
            raise ValueError('Method not recognised')

    @property
    def method(self):
        return self._method

    @method.setter
    def method(self, new_method):
        self._method = new_method