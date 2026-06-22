import numpy as np


def standardize(X):
    # assume that receive a numpy.ndarray
    means = np.mean(a=X, axis=0) # we have variables on the columns
    stds = np.std(a=X, axis=0) # the fist axis is for the columns
    return (X - means) / stds
