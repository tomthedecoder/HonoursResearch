import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from scipy.integrate import RK45
import CTRNN as ctrnn
from math import *
from Individual import Individual


def eval(t, y):
    """ Uses RK45 to approximate the two-neuron system with a single step"""

    y = [y]
    yh = []
    th = []

    solver = RK45(exponential_decay, t_bound=t, y0=y, t0=0, atol=0.1**10)

    while solver.status == "running":
        yh.append(solver.y)
        th.append(solver.t)
        solver.step()

    return yh, th


def system(Y, t, W, B, T):
    """" Two neuron system"""""

    y1 = 1/T[0] * (-Y[0] + W[0][0] * sigmoid(Y[0] + B[0]) + W[1][0] * sigmoid(Y[1] + B[1]) + I(t))
    y2 = 1/T[1] * (-Y[1] + W[0][1] * sigmoid(Y[0] + B[0]) + W[1][1] * sigmoid(Y[1] + B[1]) + I(t))

    return [y1[0], y2[0]]


def sigmoid(Y):
    return np.array([1/(1 + exp(-Y))])


def Dsigmoid(y):
    return sigmoid(y)*(1 - sigmoid(y))


def exponential_decay(t, y): return -0.5 * y


def lotkavolterra(t, z, a, b, c, d):
    x, y = z
    return [a*x - b*x*y, -c*y + d*x*y]


def I(t):
    """ The input signal with a spike"""
    normal_value = np.cos(t)
    spike_value = 10
    if 1 <= t <= 1.2:
        return normal_value #spike_value
    else:
        return normal_value


def Dy(t, y, w, tau, b):
    return 1/tau * (-y[0] + w*sigmoid(y[0] + b) + I(t))


def DJ(t, y, w, b, tau):
    return Dy(t, y, w, b, tau) * w * Dsigmoid(y + b)

# 0.17468350714525904, 1.886070207503523, 0.3531347819032027
solver1 = solve_ivp(Dy, [0, 30], [0], args=[1, 5, 5], max_step=0.01, dense_output=True)
#solver2 = solve_ivp(Dy, [0, 30], [1], args=[1, 1, 0], max_step=0.01, dense_output=True)

y0 = [0, 0]
W = [[1.54881838, 2.78710795],
     [0.0930542, -4.34517701]]
B = [2.16796316, -2.12606141]
T = [2.42980896, 0.98990346]

#yh, th = eval(y=20, t=100)

target_signal = lambda t: np.sin(t)
times = []
y = []
for idx in range(int(30/0.01)):
    times.append(idx * 0.01)
    y.append(target_signal(times[-1]))

sol = odeint(system, y0, times, args=(W, B, T))

plt.figure()
plt.plot(sol[:, 1], sol[:, 0], 'b', label='y(t)')
plt.title("Two Neuron")
plt.legend(loc='best')
plt.xlabel('y1')
plt.ylabel('y2')
plt.grid()

plt.figure()
plt.plot(times, sol[:, 1])
plt.plot(times, y)
plt.title("Last Neuron Output")
plt.xlabel("time(t)")
plt.ylabel("output(y)")
plt.grid()

plt.show()


"""plt.figure()
plt.plot(solver1.t, solver1.y[0])
#plt.plot(solver2.t, solver2.y[0])
plt.plot(times, y)
plt.legend(['Output', 'Target'])
plt.title("Single Neuron")
plt.grid()"""

"""plt.figure()
plt.plot(t, sol[:, 0], 'b', label='y1(t)')
plt.plot(t, sol[:, 1], 'g', label='y2(t)')
plt.title("Two Neuron")
plt.legend(loc='best')
plt.xlabel('t')
plt.ylabel('y')
plt.grid()

plt.figure()
plt.plot(th, yh, 'g')
plt.grid()
plt.title('Exponential Decay')
plt.xlabel('t')
plt.ylabel('y')"""

"""plt.show()"""




