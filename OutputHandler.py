from sklearn.preprocessing import minmax_scale, scale


class OutputHandler:
    def __init__(self, method='max/min'):
        """ Holds methods for scaling the output of the CTRNN's last node"""

        self._method = method.strip()

    def call(self, y_values):
        if self._method == 'max/min':
            return minmax_scale(y_values, feature_range=(-1, 1))
        elif self._method == 'white':
            return scale(y_values)
        elif self._method == 'default':
            return y_values
        else:
            raise ValueError('Method not recgongised')

    @property
    def method(self):
        return self._method

    @method.setter
    def method(self, new_method):
        self._method = new_method