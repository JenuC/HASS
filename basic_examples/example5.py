import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
from skimage import data, color, transform
import cvxpy as cp

# Load and resize a grayscale image
image = color.rgb2gray(data.astronaut())
image = transform.resize(image, (100, 100), anti_aliasing=True)

# 2D DCT and inverse DCT
def dct2(img):
    return dct(dct(img.T, norm='ortho').T, norm='ortho')

def idct2(coeffs):
    return idct(idct(coeffs.T, norm='ortho').T, norm='ortho')

# Transform image to DCT domain
dct_image = dct2(image)
x_true = dct_image.flatten()
n = x_true.size

# Sample percentages
percentages = [2, 5, 10, 25, 50, 90]
reconstructions = []
masks = []

for pct in percentages:
    print("Sampling percentage:", pct)
    # Create sampling mask
    m = int(n * pct / 100)
    sample_indices = np.random.choice(n, m, replace=False)

    A = np.random.randn(m, n)
    y = A @ x_true

    # L1 minimization with CVXPY
    x = cp.Variable(n)
    objective = cp.Minimize(cp.norm1(x))
    constraints = [A @ x == y]
    problem = cp.Problem(objective, constraints)
    problem.solve()

    # Reconstruct image from solution
    x_rec = x.value.reshape(dct_image.shape)
    recovered_image = idct2(x_rec)
    reconstructions.append(recovered_image)

    # Show DCT sampling mask
    dct_mask = np.zeros(n)
    dct_mask[sample_indices] = 1
    masks.append(dct_mask.reshape(dct_image.shape))
    

# Plot original, sampling masks, and reconstructed images
fig, axes = plt.subplots(len(percentages), 2, figsize=(8, len(percentages) * 2.5))
for i, pct in enumerate(percentages):
    axes[i, 0].imshow(masks[i], cmap='gray')
    axes[i, 0].set_title(f"{pct}% DCT Sample Mask")
    axes[i, 1].imshow(reconstructions[i], cmap='gray')
    axes[i, 1].set_title(f"Reconstructed Image ({pct}%)")
    for j in range(2):
        axes[i, j].axis('off')
plt.tight_layout()
plt.show()
