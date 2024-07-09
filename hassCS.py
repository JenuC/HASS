import pywt
import numpy as np

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

# Forward wavelet transform function
def wave(X,wavelet):
    return pywt.dwtn(X, wavelet)

# Inverse wavelet transform function
def wave_inv(X_wavelet,wavelet):
    return pywt.idwtn(X_wavelet, wavelet)

#Optimization
def shrinkage(x, alpha):
    return np.sign(x) * np.maximum(np.abs(x) - alpha, 0)

class compressed_sensing:
    def __init__(self,data,wavelet = "bior1.3",undersample_rate = .5,tau = 0.1 ,lambda_ = 0.01,max_iter = 100):
        self.data = data
        self.wavelet = wavelet
        self.tau = tau
        self.lambda_ = lambda_
        self.max_iter = max_iter
        
        shape = self.data.shape
        self.mask = create_sampling_mask(shape, undersample_rate)
        
        undersampled_data = self.data * self.mask
        self.Y = pywt.dwtn(undersampled_data, self.wavelet)
        
    #Iterative shrinking optimization
    def runCS(self):
        
        # Initialize wavelet coefficients with the same structure as Y
        X_wavelet = {key: np.zeros_like(val) for key, val in self.Y.items()}

        for k in range(self.max_iter):
            # Gradient step
            A_X_wavelet = pywt.idwtn(X_wavelet,self.wavelet) * self.mask  # Implicit sampling operation
            gradient = pywt.dwtn((A_X_wavelet - pywt.idwtn(self.Y,self.wavelet)) * self.mask,self.wavelet)  # Masked gradient

            X_wavelet = {key: X_wavelet[key] - self.tau * gradient[key] for key in X_wavelet}

            # Shrinkage step
            X_wavelet = {key: shrinkage(X_wavelet[key], self.lambda_ * self.tau) for key in X_wavelet}

        X_reconstructed = pywt.idwtn(X_wavelet,self.wavelet)
        return X_reconstructed, X_wavelet
