# Signal: sparse in frequency domain
import numpy as np
from numpy.fft import fft, ifft

n = 128
true_signal = np.zeros(n)
true_signal[[10, 20, 50]] = [1.0, -0.5, 0.75]  # Sparse spikes

# Simulate random measurements (compressed sensing)
A = np.random.randn(32, n)  # Underdetermined system
y = A @ true_signal

# Reconstruct via L1 minimization
import cvxpy as cp

x = cp.Variable(n)
objective = cp.Minimize(cp.norm1(x))  # promote sparsity
constraints = [A @ x == y]
prob = cp.Problem(objective, constraints)
result = prob.solve()

# Plot result
import matplotlib.pyplot as plt
plt.plot(true_signal, label="Original")
plt.plot(x.value, label="Reconstructed", linestyle="--")
plt.legend()
plt.title("Compressed Sensing Reconstruction")
plt.show()
