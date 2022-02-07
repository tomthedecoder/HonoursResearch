from scipy.linalg import eig
from scipy.integrate import solve_ivp
from scipy.optimize import root
from itertools import combinations
import numpy as np
from OutputHandler import *
from CTRNNParameters import *
from CTRNNStructure import *
from CTRNN import *
from plot_all_neurons import *
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def inverse_sigmoid(y):
    return np.math.log(y/(1-y), 10)


def Dsigmoid(x):
    return sigmoid(x)*(1 - sigmoid(x))


def I(t):
    return 0


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

    def inside(array, sub_array):
        for other_sub in array:
            num_same = 0
            for idx, value in enumerate(sub_array):
                if abs(value - other_sub[idx]) < 0.001:
                    num_same += 1
            if num_same == len(sub_array):
                return True

        return False

    num_nodes = len(W)
    num_iterations = 100
    tol = 0.001
    roots = np.array([])

    for _ in range(num_iterations):
        y0 = np.random.uniform(-10, 10, num_nodes)
        opti_solve = root(y_primeb, x0=y0, args=(t, W, T, B, IW, s), jac=Dy, tol=tol)
        maybe_root = opti_solve.x
        #print(maybe_root, y_prime(t, maybe_root, W, T, B, IW, s))
        if opti_solve.success and not inside(roots, maybe_root):
            if len(roots) == 0:
                roots = np.array([maybe_root])
            else:
                roots = np.append(roots, [maybe_root], 0)

    return roots


def y_nullclines(t, W, B, IW):
    """ y nullclines for the two neuron system. Will be undefined in many places, because inv_sigmoid defined
        on (0,1) onlu. Assumes CTRNN is fully connected"""

    num_nodes = len(W)

    assert num_nodes > 1

    y_range = [0, 3]
    step_size = 0.1
    num_samples = int((y_range[1] - y_range[0])/step_size)
    duration = [i * step_size for i in range(num_samples)]
    combin = list(combinations(duration, num_nodes))

    nullclines = [[] for _ in range(num_nodes)]
    domains = [[[] for _ in range(num_nodes-1)] for _ in range(num_nodes)]
    for y in combin:
        for i in range(num_nodes):
            k = i+1 if i != num_nodes-1 else 0
            sigmoid_term = 0

            for j in range(num_nodes):
                if j == k:
                    continue
                sigmoid_term += W[j][i] * sigmoid(y[j] + B[j])
                domains[k][j - (0 if j < k else 1)].append(y[j])

            term = (y[i] - sigmoid_term - IW[i] * I(t)) / W[k][i]
            if 0 > term or term > 1:
                nullclines[k].append(0)
            else:
                nullclines[k].append(inverse_sigmoid(term) - B[k])

    return np.array(domains), np.array(nullclines)


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
            sigmoid_terms[i] += W[j][i] * sigmoid(y[j] + B[j])

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
                bias = B[j] - (s/T[j])
            else:
                bias = B[j]
            sigmoid_terms[i] += W[j][i] * sigmoid(y[j] + bias)

    derivative = (-y + sigmoid_terms + IW * I(t)) / T
    derivative[-1] -= s / T[-1]

    return derivative


########################
# Parameters for CTRNN #
########################

final_t = np.ceil(6 * np.pi)
num_nodes = 3

W = np.array([[5, -1, 1],
              [1, 7, -1],
              [1, 0.0, 6]])
T = np.array([1, 2.5, 1])
B = np.array([-4.108, -2.787, -1.114])
IW = np.array([1, 1, 1])
shift = 0.0

genome = np.append(W[0], W[1])
genome = np.append(genome, W[2])
genome = np.append(genome, T)
genome = np.append(genome, B)
genome = np.append(genome, IW)
genome = np.append(genome, shift)

connection_array = [(i, j) for i in range(num_nodes) for j in range(num_nodes)]
forcing_signals = make_signals(num_nodes, [lambda t: np.sin(t)])

output_handler = OutputHandler("max/min")
params = CTRNNParameters(genome, output_handler, forcing_signals, connection_array)
ctrnn = CTRNN(params)
t_space, output = ctrnn.evaluate(final_t)

target_signal = lambda t: np.sin(2 * t)
y = target_signal(t_space)

initial_conditions = np.array([[0.0, 0.0, 0.0]])
solutions = []
for y0 in initial_conditions:
    solutions.append(solve_ivp(y_prime, t_span=(0, t_space[-1]), y0=y0, method='RK45', args=(W, T, B, IW, shift),
                                        t_eval=t_space, dense_output=True, max_step=0.01))

equilibria = root_find(0, y, W, T, B, IW, shift)
domain, nullclines = y_nullclines(0, W, B, IW)
eigen_values, eigen_vectors = eig(Dy(y, 0, W, T, B, IW, shift))

fig = plt.figure()
axs = fig.add_subplot(projection='3d')
axs.set_title("phase space")
axs.set_xlabel("y1")
axs.set_ylabel("y2")
if num_nodes == 2:
    for i, sol in enumerate(solutions):
        axs.plot(sol.y[0], sol.y[1])
    axs.plot(ctrnn.node_history[0], ctrnn.node_history[1], 'gold')
elif num_nodes == 3:
    axs.set_zlabel("y3")
    for i, sol in enumerate(solutions):
        axs.plot(sol.y[0], sol.y[1], sol.y[2])
    axs.plot(ctrnn.node_history[0], ctrnn.node_history[1], ctrnn.node_history[2], 'gold')

for point in equilibria:
    eigen_values, eigen_vectors = eig(Dy(point, 0, W, T, B, IW, shift))
    negative_counts = 0
    positive_counts = 0
    zero_counts = 0
    for lam in eigen_values:
        if np.real(lam) < 0:
            negative_counts += 1
        elif np.real(lam) > 0:
            positive_counts += 1
        elif np.real(lam) == 0:
            zero_counts += 1

    # assign color based on type
    # saddle
    if negative_counts > 0 and positive_counts > 0:
        color = "black"
    # sink
    elif negative_counts > 0:
        color = "blue"
    # source
    elif positive_counts > 0:
        color = "red"
    # unknown
    else:
        color = "orange"

    axs.plot(point[0], point[1], point[2], 'o', color=color)
    print(f"equilibria at {point} has eigen values {eigen_values}")

plot_eigen_vectors = False
if plot_eigen_vectors:
    initial = equilibria[0]
    other_t_space = np.linspace(-3, 3, 100)
    principle_axis = [[] for _ in range(num_nodes ** 2)]
    for t in other_t_space:
        e1 = initial + t * eigen_vectors[0]
        e2 = initial + t * eigen_vectors[1]
        c = 0
        for i in range(0, len(principle_axis), num_nodes):
            eig_vec = initial + t * eigen_vectors[c]
            principle_axis[i].append(eig_vec[0])
            principle_axis[i + 1].append(eig_vec[1])
            if num_nodes == 3:
                principle_axis[i + 2].append(eig_vec[2])
            c += 1

    for i in range(0, len(principle_axis), num_nodes):
        y1 = principle_axis[i]
        y2 = principle_axis[i + 1]
        if num_nodes == 3:
            y3 = principle_axis[i + 2]
            axs.plot(y1, y2, y3, "--")
        else:
            axs.plot(y1, y2, "--")

axs.grid()
axs.legend(["solution #1", "solution #2", "solution #3", "ctrnn"])

fig = plt.figure()
axs = fig.add_subplot(projection='3d')
# synaptic input space nullclines for num_nodes == 2
if num_nodes == 2:
    plt.title("J Nullclines")
    plt.xlabel("J1")
    plt.ylabel("J2")
    plt.legend(["J1", "J2"])
    #plt.plot(J1, nullclines[0])
    #plt.plot(nullclines[1], J2)
elif num_nodes == 3:
    axs.set_title("Y Nullclines")
    axs.set_xlabel("y1")
    axs.set_ylabel("y2")
    axs.set_zlabel("y3")
    axs.scatter(nullclines[0], domain[0][0], domain[0][1], marker='o')
    axs.scatter(domain[1][0], nullclines[1], domain[1][1], marker='^')
    axs.scatter(domain[2][0], domain[2][1], nullclines[2], marker='x')

plot_all_neurons(ctrnn, final_t)

fig = plt.figure()
axs = fig.add_subplot()
for sol in solutions:
    axs.plot(t_space, sol.y[-1], '--')

legend = [f"solution{i}" for i in range(len(solutions))]
legend.append("ctrnn")
legend.append("target")

axs.plot(t_space, output, '--')
axs.plot(t_space, y)
axs.set_title("Last Neuron Output")
axs.set_xlabel("time(t)")
axs.set_ylabel("output(y)")
axs.legend(legend)
axs.grid()


plt.show()








