import numpy as np


def replaceNAN(X):
    '''
    replace NaN (not a number) or
    NA (not available, not applicable)
    cells with the mean (variable) on the column
    X - expect a numpy.ndarray
    '''

    avgs = np.nanmean(X, axis=0)
    pos = np.where(np.isnan(X))
    # print(pos, type(pos))
    X[pos] = avgs[pos[1]]
    return X


def standardise(X):
    '''
    X - expect a numpy.ndarray
    '''
    avgs = np.mean(X, axis=0)
    stds = np.std(X, axis=0)
    Xstd = (X - avgs) / stds
    return Xstd