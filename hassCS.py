import pywt
import numpy as np

#Taken from https://github.com/oliviaguest/gini/blob/master/gini.py
def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    # All values are treated equally, arrays must be 1d:
    array = array.flatten()
    if np.amin(array) < 0:
        # Values cannot be negative:
        array -= np.amin(array)
    # Values cannot be 0:
    array += 0.0000001
    # Values must be sorted:
    array = np.sort(array)
    # Index per array element:
    index = np.arange(1,array.shape[0]+1)
    # Number of array elements:
    n = array.shape[0]
    # Gini coefficient:
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)))

#Having the mask allows for easier matrix math and less reliance on pywt
def create_sampling_mask(shape, sampling_rate):
    return np.random.rand(*shape) < sampling_rate

#Optimization
def shrinkage(x, alpha):
    return np.sign(x) * np.maximum(np.abs(x) - alpha, 0)

#Iterative shrinking optimization
def runCS(data,wavelet = "bior1.3",undersample_rate = .5,tau = 0.1 ,lambda_ = 0.01,max_iter = 100):

    shape = data.shape
    mask = create_sampling_mask(shape, undersample_rate)

    undersampled_data = data * mask
    Y = pywt.dwtn(undersampled_data, wavelet)

    # Initialize wavelet coefficients with the same structure as Y
    X_wavelet = {key: np.zeros_like(val) for key, val in Y.items()}

    for k in range(max_iter):
        # Gradient step
        A_X_wavelet = pywt.idwtn(X_wavelet,wavelet) * mask  # Implicit sampling operation
        gradient = pywt.dwtn((A_X_wavelet - pywt.idwtn(Y,wavelet)) * mask, wavelet)  # Masked gradient

        X_wavelet = {key: X_wavelet[key] - tau * gradient[key] for key in X_wavelet}

        # Shrinkage step
        X_wavelet = {key: shrinkage(X_wavelet[key], lambda_ * tau) for key in X_wavelet}

    X_reconstructed = pywt.idwtn(X_wavelet,wavelet)
    return X_reconstructed, X_wavelet
