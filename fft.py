import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq
from read_from_file import *



#y_output = read_from_file("output_file")
#y_target = read_from_file("targets_file")
#times = read_from_file("time_file")

target_signal = lambda t : np.sin(t)
output_signal = lambda t : np.cos(t)

y_output = []
y_target = []
times = []
duration = 2 * np.pi
DT = 0.01
for idx in range(int(duration/DT)):
    times.append(DT * idx)
    y_output.append(output_signal(times[-1]))
    y_target.append(target_signal(times[-1]))


# Number of sample points
N = 800

# sample spacing
T = 1.0 / N

fourier_of_output = fft(y_output)
fourier_of_target = fft(y_target)
tf = fftfreq(N, T)[:N//2]

plt.figure()
plt.title("Target And Output")
plt.grid()
plt.plot(times, y_output)
plt.plot(times, y_target)
plt.legend(["Output", "Target"])

plt.figure()
plt.title("Fourier Transforms")
plt.grid()
plt.plot(tf, 1.0/N * np.abs(fourier_of_output[0:N//2]), 'g')
plt.plot(tf, 1.0/N * np.abs(fourier_of_target[0:N//2]), 'orange')
plt.legend(["Output", "Target"])

plt.show()
