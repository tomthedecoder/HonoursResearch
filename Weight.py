class Weight:
    def __init__(self, i, j, value):
        """ Container class to keep CTRNN.calculate_derivative O(n^2), consider the weight matrix doesn't need to be
            stored and later code can just iterate over self.genome, making a O(self.num_weights) calculation
            per evaluation"""

        self.i = i
        self.j = j
        self.value = value

    def traveled_from(self):
        return self.i

    def traveled_to(self):
        return self.j

    def __str__(self):
        return "({},{},{})".format(self.i, self.j, self.value)