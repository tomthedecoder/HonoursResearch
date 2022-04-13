from sklearn.preprocessing import scale, maxabs_scale, minmax_scale
import numpy as np


def mean_shift(y_values):
    """ Scaling method which sends x to x - mu"""

    start_point = int(10 * len(y_values) / 100)
    mu = np.mean(y_values[start_point:])

    return np.array([y - mu for y in y_values])


def origin_shift(y_values):
    """ shifts an array to start at the origin"""

    start_value = y_values[0]

    return np.array([y - start_value for y in y_values])


def scale_shift_to_origin(y_values):
    """"" Shifts array to origin and scales values to unit size"""

    start_value = y_values[0]

    return np.array([100*(y - start_value) for y in y_values])


def range_scale(values, low=-1, high=1):
        values = values - values[0]

        minimum = min(values)
        maximum = max(values)

        """if maximum < high:
            high = maximum
        if minimum > low:
            low = minimum

        new_values = []
        for i, x in enumerate(values):
            x_new = (high - low) * ((x - minimum) / (maximum - minimum)) + low
            new_values.append(x_new)"""

        new_values = scale(values)

        return new_values - new_values[0]


class OutputHandler:
    def __init__(self, method='max/min'):
        """ Scales the output of the CTRNN's last node"""

        self._method = method.strip()

    def call(self, values):
        if self._method == 'max/min':
            return minmax_scale(values, feature_range=(-1, 1))
        elif self._method == 'range':
            return range_scale(values)
        elif self._method == 'scale':
            return scale(values)
        elif self._method == 'origin':
            return origin_shift(values)
        elif self._method == 'scale&origin':
            return scale_shift_to_origin(values)
        elif self._method == 'white':
            return scale(values)
        elif self._method == 'maxabs':
            return maxabs_scale(values)
        elif self._method == 'mean shift':
            return mean_shift(values)
        elif self._method == 'default':
            return values
        else:
            raise ValueError('Method not recognised')

    @property
    def method(self):
        return self._method

    @method.setter
    def method(self, new_method):
        self._method = new_method