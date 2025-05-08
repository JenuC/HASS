
from scipy.fftpack import dct, idct
import cvxpy as cp
import numpy as np
def process_image(image):
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
    
    return recovered_image
