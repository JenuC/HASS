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

#l1-norm 
def shrinkage(x, alpha):
    return np.sign(x) * np.maximum(np.abs(x) - alpha, 0)
def soft_thresh(x, lam):
    if ~(isinstance(x[0], complex)):
        return np.zeros(x.shape) + (x + lam) * (x<-lam) + (x - lam) * (x>lam) 
    else:
        return np.zeros(x.shape) + ( abs(x) - lam ) / abs(x) * x * (abs(x)>lam) 


def flat_wavelet_transform2(x, method='bior1.3'):
    """Wavelet transform using wavedecn, returns coefficients as a list."""
    coeffs = pywt.wavedecn(x, method)
    flat_coeffs = np.concatenate([c.flatten() for c in pywt.coeffs_to_array(coeffs)[0]])
    return flat_coeffs, coeffs  # Return coeffs for inverse transform


def inverse_flat_wavelet_transform2(X, coeffs_structure, method='bior1.3'):
    """Inverse wavelet transform using waverecn, reshapes coefficients into original structure."""
    coeffs_array, slices = pywt.coeffs_to_array(coeffs_structure)
    coeffs_array_flat = np.copy(coeffs_array)
    coeffs_array_flat[:] = X[:coeffs_array.size].reshape(coeffs_array.shape)
    new_coeffs = pywt.array_to_coeffs(coeffs_array_flat, slices, output_format='wavedecn')
    return pywt.waverecn(new_coeffs, method)


#Iterative shrinking optimization
def single_rec(data,wavelet = "bior1.3",undersample_rate = .5,tau = 0.995 ,lambda_ = 0.01,max_iter = 50):
    
    #This step creates the phi sensing basis
    shape = data.shape
    mask = create_sampling_mask(shape, undersample_rate)

    #This step creates our y, aka our measured data in the sparse wavelet domain
    undersampled_data = data * mask
    Y = pywt.dwtn(undersampled_data, wavelet)

    # Initialize wavelet coefficients with the same structure as Y, pywavelets uses dictionaries
    X_wavelet = {key: np.zeros_like(val) for key, val in Y.items()}
    #Optimization 
    for k in range(max_iter):
        # Gradient step
        A_X_wavelet = pywt.idwtn(X_wavelet,wavelet)  # Implicit sampling operation
        gradient = pywt.dwtn((A_X_wavelet - pywt.idwtn(Y,wavelet)), wavelet)  # Masked gradient
        
        X_wavelet = {key: X_wavelet[key] - tau * gradient[key] for key in X_wavelet}

        # Shrinkage step: Apply soft-thresholding to enforce sparsity (L1 regularization)
        X_wavelet = {key: shrinkage(X_wavelet[key], lambda_ * tau) for key in X_wavelet}

        
    X_reconstructed = pywt.idwtn(X_wavelet,wavelet)


    return X_reconstructed, X_wavelet


def multi_rec(y, method='sym3', lam=100, lam_decay=0.995, max_iter=80):
    """
    Performs iterative wavelet reconstruction with soft thresholding.

    Parameters:
        y (ndarray): The undersampled data.
        shape (tuple): Shape of the original image.
        method (str): Wavelet transform method (default is 'sym3').
        lam (float): Initial thresholding parameter (default is 100).
        lam_decay (float): Decay rate of the lambda parameter (default is 0.995).
        max_iter (int): Maximum number of iterations (default is 80).

    Returns:
        xhat (ndarray): The reconstructed image.
    """
    shape = y.shape
    
    xhat = y.copy()
    for i in range(max_iter):
        xhat_old = xhat
        
        # Wavelet transform
        Xhat_old, coeffs_structure = flat_wavelet_transform2(xhat, method)
        
        # Apply soft-thresholding
        Xhat = soft_thresh(Xhat_old, lam)
        
        # Inverse wavelet transform
        xhat = inverse_flat_wavelet_transform2(Xhat, coeffs_structure, method)
        
        # Enforce known undersampled values
        xhat[y != 0] = y[y != 0]
        
        # Clip values to valid image range
        xhat = xhat.astype(int)
        xhat = np.clip(xhat, 0, 255)

        # Decay lambda for next iteration
        lam *= lam_decay
    
    return xhat

