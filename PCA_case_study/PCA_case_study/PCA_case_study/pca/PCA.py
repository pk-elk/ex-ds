'''
A class for implementing PCA (Principal Components Analysis)
'''
import numpy as np
from scipy.signal import ellip

from PCA_case_study import Functions as fun


class PCA:
    def __init__(self, X):
        self.X = X
        # compute the correlation matrix
        self.corr = np.corrcoef(x=self.X, rowvar=False) # we have the variables on the columns
        # standardize the observed variables
        self.X_std = fun.standardize(self.X)
        # compute the variance-covariance matrix on X_std
        self.cov = np.cov(m=self.X_std, rowvar=False) # variables are on columns
        # extract the eigenvalues and the eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(a=self.cov)
        print(type(eigenvectors), eigenvectors.shape)
        print(eigenvalues, type(eigenvalues), eigenvalues.shape)
        k_desc = [k for k in reversed(np.argsort(a=eigenvalues))]
        print(k_desc)
        self.alpha = eigenvalues[k_desc]
        self.a = eigenvectors[:, k_desc]
        # regularization of eigenvectors
        for j in range(len(self.alpha)):
            min = np.min(self.a[:, j])
            max = np.max(self.a[:, j])
            if np.abs(min) > np.abs(max):
                self.a[:, j] = -self.a[:, j]
                
        # compute the principal components
        self.C = self.X_std @ self.a

        # compute the factor loadings
        # the correlation between the observed variables and the principal components
        self.Rxc = self.a * np.sqrt(self.alpha)



    def getCorr(self):
        return self.corr

    def getStd(self):
        return self.X_std

    def getAlpha(self):
        return self.alpha

    def getComponents(self):
        return self.C

    def getFactorLoadings(self):
        return self.Rxc

    def getScores(self):
        return self.C / np.sqrt(self.alpha)

    def getCommunalities(self):
        # Rxc2 = self.Rxc * self.Rxc
        Rxc2 = np.square(self.Rxc)
        return np.cumsum(a=Rxc2, axis=1) # we compute the cumulative cums on the lines