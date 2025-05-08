import numpy as np

from scipy.fftpack import dct, idct
import cvxpy as cp
from skimage import data, color, transform

import matplotlib.pyplot as plt


# Load a grayscale image and downscale it to 100x100
image = color.rgb2gray(data.astronaut())
image = transform.resize(image, (100, 100), anti_aliasing=True)

# 2D Discrete Cosine Transform (DCT) and inverse
def dct2(img):
    return dct(dct(img.T, norm='ortho').T, norm='ortho')

def idct2(coeffs):
    return idct(idct(coeffs.T, norm='ortho').T, norm='ortho')

# Transform image to DCT domain
dct_image = dct2(image)
x_true = dct_image.flatten()
n = x_true.size

# Simulate compressed sensing with 20% random measurements
m = int(n * 0.2)
A = np.random.randn(m, n)
y = A @ x_true

# L1 recovery using CVXPY
x = cp.Variable(n)
objective = cp.Minimize(cp.norm1(x))
constraints = [A @ x == y]
problem = cp.Problem(objective, constraints)
problem.solve()

# Reshape and reconstruct image
x_rec = x.value.reshape(dct_image.shape)
recovered_image = idct2(x_rec)

# Plot results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(image, cmap='gray')
axes[0].set_title("Original Image")
axes[1].imshow(np.log(np.abs(dct_image) + 1e-3), cmap='gray')
axes[1].set_title("DCT Coefficients (log scale)")
axes[2].imshow(recovered_image, cmap='gray')
axes[2].set_title("Recovered Image (20% Samples)")
for ax in axes:
    ax.axis('off')
plt.tight_layout()
plt.show()
