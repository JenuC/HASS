import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
from skimage import data, color, transform
import cvxpy as cp
import time

# Load and resize grayscale image
image = color.rgb2gray(data.astronaut())
image = transform.resize(image, (100, 100), anti_aliasing=True)

# 2D DCT and inverse
def dct2(img):
    return dct(dct(img.T, norm='ortho').T, norm='ortho')
def idct2(coeffs):
    return idct(idct(coeffs.T, norm='ortho').T, norm='ortho')

# Transform to DCT domain
dct_image = dct2(image)
x_true = dct_image.flatten()
n = x_true.size

# Percentages to test
percentages = [2, 5, 10, 20, 30, 40, 50]
truncate_times, random_times, cs_times = [], [], []

for pct in percentages:
    m = int(n * pct / 100)
    print("Sampling percentage:", pct)
    # Top-K DCT (truncate)
    start = time.time()
    top_indices = np.argsort(np.abs(x_true))[-m:]
    mask = np.zeros(n)
    mask[top_indices] = 1
    truncate_dct = x_true * mask
    _ = idct2(truncate_dct.reshape(dct_image.shape))
    truncate_times.append(time.time() - start)

    # Random mask (zeroed)
    start = time.time()
    rand_indices = np.random.choice(n, m, replace=False)
    mask = np.zeros(n)
    mask[rand_indices] = 1
    random_dct = x_true * mask
    _ = idct2(random_dct.reshape(dct_image.shape))
    random_times.append(time.time() - start)

    # Compressed sensing (CVXPY)
    start = time.time()
    A = np.random.randn(m, n)
    y = A @ x_true
    x = cp.Variable(n)
    objective = cp.Minimize(cp.norm1(x))
    constraints = [A @ x == y]
    cp.Problem(objective, constraints).solve()
    cs_times.append(time.time() - start)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(percentages, truncate_times, label="Top-K DCT (Truncate)", marker='o')
plt.plot(percentages, random_times, label="Random DCT (Zeroed)", marker='o')
plt.plot(percentages, cs_times, label="Compressed Sensing (CVXPY)", marker='o')
plt.xlabel("Percentage of DCT Data Used")
plt.ylabel("Computation Time (seconds)")
plt.title("Computation Time vs Data Used")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
