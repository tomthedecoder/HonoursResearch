import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import RK45
from scipy.optimize import fsolve
import itertools


def J_prime(x, t, *args):
    J1 = x[0]
    J2 = x[1]

    J1p = (1/args[0][1]) * (J1 - np.divide(np.pow(J1, 2), args[1][1][0])) * (np.divide(args[1][1][1], args[1][1][0]) * J1 - np.log10(np.divide(J1, args[1][1][0] - J1)) + J2 + args[3][1] + args[4][1])
    J2p = (1/args[0][0]) * (J2 - np.divide(np.pow(J2, 2), args[1][0][1])) * (np.divide(args[1][0][0], args[1][0][1]) * J2 - np.log10(np.divide(J2, args[1][0][1] - J2)) + J1 + args[3][0] + args[4][1])

    return J1p, J2p


def y_nullcline(domain, params):
    nullcline_1 = []
    nullcline_2 = []
    for J in domain:
        y1 =   n
        nullcline_1.append(np.log10(y1/(1 - y1)) - params[3][1])
        nullcline_2.append(np.log10(np.divide(J, params[1][0][1] - J)) - np.divide(params[1][0][0], params[1][0][1]) * J - params[2][0] - params[3][0])

    return domain, nullcline_1, nullcline_2


def J_nullcline(domain, params):
    nullcline_1 = []
    nullcline_2 = []
    for J in domain:
        nullcline_1.append(np.log10(np.divide(J, params[1][1][0] - J)) - np.divide(params[1][1][1], params[1][1][0]) * J - params[2][1] - params[3][1])
        nullcline_2.append(np.log10(np.divide(J, params[1][0][1] - J)) - np.divide(params[1][0][0], params[1][0][1]) * J - params[2][0] - params[3][0])

    return domain, nullcline_1, nullcline_2


def solve_nullclines(null_equation, params: tuple, dim: int, start: float, end: float):
    assert callable(null_equation)

    step_size = 0.01

    # populate partition array
    domain = [x for x in np.linspace(start, end, int(np.divide(end - start, step_size)), endpoint=False)]

    # call equation to solve
    return null_equation(domain, params)


def plot_nullclines():

    # range nullclines are calculated over
    x_low = 0.01
    x_high = 1

    # init some parameters
    weights = np.array([[6, 1], [1, 3]])
    taus = np.array([1, 1])
    biases = np.array([-3.4, -2])
    inputs = np.array([0, 0])

    params = (taus, weights, inputs, biases)

    # call oracle for nullclines
    J, J1_null, J2_null = solve_nullclines(J_nullcline, params, 2, x_low, x_high)

    # plot nullclines
    plt.figure()
    plt.title("J-Space Nullclines")
    plt.grid()
    plt.xlabel("J1")
    plt.ylabel("J2")
    plt.plot(J, J1_null, '-')
    plt.plot(J, J2_null, '-')

    plt.show()


plot_nullclines()