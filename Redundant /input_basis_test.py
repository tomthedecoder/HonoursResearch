import matplotlib.pyplot as plt
import numpy as np

signal = lambda x: np.exp(-0.1 * x) * np.sin(x)


def sigmoid(Y):
    return 1/(1 + np.exp(-Y))


def system(Y, W, B, T, I):
    """" Two neuron system"""
    y1 = 1/T[0] * (-Y[0] + W[0] * sigmoid(Y[0] + B[0]) + I[0])
    #y2 = 1/T[1] * (-Y[1] + W[0][1] * sigmoid(Y[0] + B[0]) + W[1][1] * sigmoid(Y[1] + B[1]) + I)

    return y1


def update(y_diff, y, DT):
    return y + y_diff * DT


def I(t, a, b, d, e, w1, w2):
    """ input to some neuron"""
    return np.exp(-0.05 * t) * np.cos(t)


# neuron parameters
y = [1]
W = [-10, 3]
B = [-5]
T = [5]

# input parameters
a = 5
b = 1
w1 = 0.5
w2 = 0.5
d = 1
e = 1

times = []
input_y = []
actual_outputs = []
pred_outputs = []

final_t = 25
step_size = 0.1

for n in range(int(final_t / step_size)):
    times.append(step_size * n)
    input_y.append(I(times[-1], a, b, d, e, w1, w2))

    actual_outputs.append(signal(times[-1]))

    y_diff = system(y, W, B, T, [input_y[-1]])
    pred_outputs.append(update(y_diff, y[0], step_size))
    y = [pred_outputs[-1]]

plt.figure()
plt.grid()
plt.title("Neuron Input")
plt.xlabel("Time(t)")
plt.ylabel("Output(y)")
#plt.plot(times, input_y, "g")
plt.plot(times, actual_outputs, "r")
plt.plot(times, pred_outputs)
plt.legend(["forcing", "target", "output"])

plt.show()