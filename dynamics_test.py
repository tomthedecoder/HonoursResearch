import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def I(t):
    return np.sin(t)


def Dsigmoid(x):
    return sigmoid(x)*(1 - sigmoid(x))


def two_neuron(t, y, W, T, B):
    """" Two neuron system"""""

    y1 = 1/T[0] * (-y[0] + W[0][0] * sigmoid(y[0] + B[0]) + W[1][0] * sigmoid(y[1] + B[1]) + I(t))
    y2 = 1/T[1] * (-y[1] + W[0][1] * sigmoid(y[0] + B[0]) + W[1][1] * sigmoid(y[1] + B[1]) + I(t))

    return [y1, y2]


def one_neuron(t, y, w, tau, b):
    return 1/tau * (-y + w*sigmoid(y + b) + I(t))


def one_neuron_synaptic(t, y, w, b, tau):
    return one_neuron(t, y, w, b, tau) * w * Dsigmoid(y + b)


y0 = [0, 0]
W = [[1, 0.42561895337358324],
     [-1, -1.676084230734908]]
B = [0.9468674242841031, 0.5911299159988552]
T = [0.485746038528897, 0.03630363856686625]

t_space = np.linspace(0, 12 * np.pi, 4000)
target_signal = lambda t: np.sin(t)
y = target_signal(t_space)

#sol = solve_ivp(two_neuron, t_span=(0, t_space[-1]), y0=[0, 0], method='RK45', args=(W, T, B), t_eval=t_space, dense_output=True, max_step=0.01)
sol = solve_ivp(one_neuron, t_span=(0, t_space[-1]), y0=[0], method='RK45', args=(1, 5, 2), t_eval=t_space, dense_output=True, max_step=0.01)

"""plt.figure()
plt.plot(sol.y[0], sol.y[1], 'b', label='y(t)')
plt.title("Two Neuron")
plt.legend(loc='best')
plt.xlabel('y1')
plt.ylabel('y2')
plt.grid()"""

plt.figure()
plt.plot(t_space, sol.y[0])
plt.plot(t_space, y)
plt.title("Last Neuron Output")
plt.xlabel("time(t)")
plt.ylabel("output(y)")
plt.legend(["output", "target"])
plt.grid()

plt.show()


plt.figure()
plt.plot(solver1.t, solver1.y[0])
#plt.plot(solver2.t, solver2.y[0])
plt.plot(times, y)
plt.legend(['Output', 'Target'])
plt.title("Single Neuron")
plt.grid()





