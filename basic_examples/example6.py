import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
from skimage import data, color, transform
import cvxpy as cp

# Load and preprocess image
image = color.rgb2gray(data.astronaut())
image = transform.resize(image, (100, 100), anti_aliasing=True)

# DCT and inverse DCT
def dct2(img):
    return dct(dct(img.T, norm='ortho').T, norm='ortho')
def idct2(coeffs):
    return idct(idct(coeffs.T, norm='ortho').T, norm='ortho')

# Full DCT of the image
dct_image = dct2(image)
x_true = dct_image.flatten()
n = x_true.size
m = int(n * 0.2)  # 20% of data

# --- Method 1: Top 20% DCT Coefficients (Truncate) ---
flat_dct = x_true.copy()
top_indices = np.argsort(np.abs(flat_dct))[-m:]
truncate_mask = np.zeros(n)
truncate_mask[top_indices] = 1
truncate_dct = flat_dct * truncate_mask
truncate_image = idct2(truncate_dct.reshape(dct_image.shape))

# --- Method 2: Random 20% DCT Coefficients (Zeroed) ---
rand_indices = np.random.choice(n, m, replace=False)
random_mask = np.zeros(n)
random_mask[rand_indices] = 1
random_dct = flat_dct * random_mask
random_image = idct2(random_dct.reshape(dct_image.shape))

# --- Method 3: Compressed Sensing via CVXPY ---
A = np.random.randn(m, n)
y = A @ x_true
x = cp.Variable(n)
objective = cp.Minimize(cp.norm1(x))
constraints = [A @ x == y]
problem = cp.Problem(objective, constraints)
problem.solve()
cs_dct = x.value.reshape(dct_image.shape)
cs_image = idct2(cs_dct)

# --- Plot all methods ---
titles = [
    "Original Image",
    "Top 20% DCT (Truncate)",
    "Random 20% DCT (Zeroed)",
    "Compressed Sensing (CVXPY)"
]
images = [image, truncate_image, random_image, cs_image]

fig, axes = plt.subplots(1, 4, figsize=(16, 5))
for ax, img, title in zip(axes, images, titles):
    ax.imshow(img, cmap='gray')
    ax.set_title(title)
    ax.axis('off')
plt.tight_layout()
plt.show()
