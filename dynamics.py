import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from CTRNN import CTRNN


def two_neuron(t, y, W, T, B, IW):
    """" Two neuron system"""""

    y1 = (-y[0] + W[0][0] * sigmoid(y[0] + B[0]) + W[1][0] * sigmoid(y[1] + B[1]) + IW[0] * I(t)) / T[0]
    y2 = (-y[1] + W[0][1] * sigmoid(y[0] + B[0]) + W[1][1] * sigmoid(y[1] + B[1]) + IW[1] * I(t)) / T[1]

    return [y1, y2]


def one_neuron(t, y, w, tau, b, iw, g):
    return (-y + w*sigmoid(g*(y + b)) + iw*I(t)) / tau


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def inverse_sigmoid(y):
    return np.math.log(y/(1-y), 10)


def I(t):
    return 0 #np.sin(t)


def Dsigmoid(x):
    return sigmoid(x)*(1 - sigmoid(x))


def J_prime(t, J, W, T, B, IW):
    J1 = (J[0] - np.power(J[0], 2)/W[1][0]) * ((W[1][1]/W[1][0]) * J[0] - np.math.log(J[0]/(W[1][0] - J[0]), 10) + J[1] + IW[1] * I(t) + B[1]) / T[1]
    J2 = (J[1] - np.power(J[1], 2)/W[0][1]) * ((W[0][0]/W[0][1]) * J[1] - np.math.log(J[1]/(W[0][1] - J[1])) + J[0] + IW[0] * I(t) + B[0]) / T[0]
    return [J1, J2]


def J_nullclines(t, W, B):
    step_size = 0.0001
    J_range = [step_size, 1 - step_size]
    num_samples = int((J_range[1] - J_range[0])/step_size)
    J1 = [i * step_size for i in range(1, num_samples+1)]
    J2 = [-i * step_size for i in range(1, num_samples+1)]

    nullclines = [[] for _ in range(2)]

    for jv in J1:
        nullclines[0].append(np.math.log(jv / (W[1][0] - jv), 10) - (W[1][1] / W[1][0]) * jv - IW[1] * I(t) - B[1])

    for jv in J2:
        nullclines[1].append(np.math.log(jv / (W[0][1] - jv), 10) - (W[0][0] / W[0][1]) * jv - IW[0] * I(t) - B[0])

    return J1, J2, nullclines


def root_find(t, y, W, T, B, IW, s):
    num_nodes = len(W)
    num_iterations = 100
    tol = 0.1
    roots = np.array([])

    for _ in range(num_iterations):
        y0 = np.random.uniform(-4, 4, num_nodes)
        maybe_root = fsolve(y_primeb, x0=y0, args=(t, W, T, B, IW, s), fprime=Dy)
        if np.sum([abs(y[idx]) for idx in range(num_nodes)]) < tol:
            if len(roots) == 0:
                roots = np.array([maybe_root])
            else:
                roots = np.append(roots, [maybe_root], 0)

    return roots


def y_nullclines(t, W, B, IW):
    """ y nullclines for the two neuron system. Will be undefined in many places, because inv_sigmoid defined
        on (0,1) onlu"""

    y_range = [-2, 2]
    step_size = 0.05
    duration = int((y_range[1] - y_range[0])/step_size)
    N = len(W)
    Y = []
    for idx in range(duration+1):
        y_value = y_range[0] + idx * step_size
        Y.append((y_value, y_value))

    nullclines = [[] for _ in range(N)]
    domains = [[] for _ in range(N)]
    undefined = []
    for y in Y:
        for i in range(N):
            k = i+1 if i != N-1 else 0
            term = (y[i] - np.sum([W[j][i] * sigmoid(y[j] + B[j]) if j != k else 0.0 for j in range(N)]) - IW[i] * I(t)) / W[k][i]
            if 0 >= term or term >= 1:
                undefined.append(y)
                continue
            domains[k].append(y[k])
            nullclines[k].append(inverse_sigmoid(term) - B[k])

    return nullclines, domains, undefined


def Dy(y, t, W, T, B, IW, s):
    """ Jaccobian of y_prime"""

    n = len(W)
    jacc = [[(-1 + W[i][i] * Dsigmoid(y[i] + B[i]))/T[i] if i == j else (W[j][i] * Dsigmoid(y[j] + B[j]))/T[i] for j in range(n)] for i in range(n)]
    return np.array(jacc)


def y_prime(t, y, W, T, B, IW, s):
    """ Simulates output for an n neuron network"""

    num_nodes = len(W)
    sigmoid_terms = np.array([0.0 for _ in range(len(W))])
    for i in range(num_nodes):
        for j in range(num_nodes):
            # shift in coordinates
            if j == num_nodes - 1:
                bias = B[j] - s/T[j]
            else:
                bias = B[j]
            sigmoid_terms[i] += W[j][i] * sigmoid(y[j] + bias)

    derivative = (-y + sigmoid_terms + IW * I(t)) / T
    derivative[-1] -= s/T[-1]

    return derivative


def y_primeb(y, t, W, T, B, IW, s):
    """ Simulates output for an n neuron network"""

    num_nodes = len(W)
    sigmoid_terms = np.array([0.0 for _ in range(len(W))])
    for i in range(num_nodes):
        for j in range(num_nodes):
            # shift in coordinates
            if j == num_nodes - 1:
                bias = B[j] - s / T[j]
            else:
                bias = B[j]
            sigmoid_terms[i] += W[j][i] * sigmoid(y[j] + bias)

    derivative = (-y + sigmoid_terms + IW * I(t)) / T
    derivative[-1] -= s / T[-1]

    return derivative


final_t = np.ceil(12 * np.pi)
num_nodes = 2

W = np.array([[2, 0],
              [0, 2]])
T = np.array([1, 1])
B = np.array([-1, -1])
IW = np.array([1, 1])
shift = 0.4

"""W = np.array([[1]])
T = np.array([1])
B = [1]
IW = [1]"""

genome = np.append(W[0], W[1])
genome = np.append(genome, T)
genome = np.append(genome, B)
genome = np.append(genome, IW)
genome = np.append(genome, shift)

connection_array = []
for i in range(1, num_nodes+1):
    for j in range(1, num_nodes+1):
        connection_array.append((i, j))

ctrnn = CTRNN(num_nodes, genome, connection_array)
t_space, output = ctrnn.evaluate(final_t)

target_signal = lambda t: np.sin(2 * t)
y = target_signal(t_space)

ya = np.array([1, 2])
yb = np.array([6, 0])
yc = np.array([0, 0])

sola = solve_ivp(y_prime, t_span=(0, t_space[-1]), y0=ya, method='RK45', args=(W, T, B, IW, shift), t_eval=t_space, dense_output=True, max_step=0.01)
solb = solve_ivp(y_prime, t_span=(0, t_space[-1]), y0=yb, method='RK45', args=(W, T, B, IW, shift), t_eval=t_space, dense_output=True, max_step=0.01)
solc = solve_ivp(y_prime, t_span=(0, t_space[-1]), y0=yc, method='RK45', args=(W, T, B, IW, shift), t_eval=t_space, dense_output=True, max_step=0.01)

equilibria = root_find(0, y, W, T, B, IW, shift)

#J1, J2, nullclines = J_nullclines(0, W, B)

plt.figure()
plt.title("phase space")
plt.xlabel("y1")
plt.ylabel("y2")
plt.plot(sola.y[0], sola.y[1], 'royalblue')
plt.plot(solb.y[0], solb.y[1], 'lightsteelblue')
plt.plot(solc.y[0], solc.y[1], 'cornflowerblue')
plt.plot(ctrnn.node_history[0], ctrnn.node_history[1], 'gold')

for point in equilibria:
    plt.plot(point[0], point[1], 'o')

eigen_values, eigen_vectors = eig(Dy(y, 0, W, T, B, IW, shift))

print('eigen values', eigen_values)

plot_eigen_vectors = True
if plot_eigen_vectors:
    initial = equilibria[0]
    other_t_space = np.linspace(-3, 3, 100)
    principle_axis = [[], [],
                      [], []]
    for t in other_t_space:
        e1 = initial + t * eigen_vectors[0]
        e2 = initial + t * eigen_vectors[1]
        principle_axis[0].append(e1[0])
        principle_axis[1].append(e1[1])
        principle_axis[2].append(e2[0])
        principle_axis[3].append(e2[1])

    for idx in range(0, len(principle_axis), 2):
        y1 = principle_axis[idx]
        y2 = principle_axis[idx + 1]
        plt.plot(y1, y2, "--")

plt.grid()
plt.legend(["solution #1", "solution #2", "solution #3", "ctrnn"])

plt.figure()
plt.title("J Nullclines")
plt.xlabel("J1")
plt.ylabel("J2")
plt.legend(["J1", "J2"])
#plt.plot(J1, nullclines[0])
#plt.plot(nullclines[1], J2)
plt.legend(["J1", "J2"])

plt.figure()
plt.plot(t_space, solc.y[-1], '--')
#plt.plot(t_space, sol2.y[-1], '--')
plt.plot(t_space, output, '--')
plt.plot(t_space, y)
plt.title("Last Neuron Output")
plt.xlabel("time(t)")
plt.ylabel("output(y)")
plt.legend(["n neuron", "ctrnn", "target"])
plt.grid()

plt.show()








