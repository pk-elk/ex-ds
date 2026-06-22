import numpy as np
import utilsCCA as utils
import scipy.stats as sts
import sklearn.preprocessing as pp


class CCA:

    def __init__(self, X, Y):

        # Canonical Correlation Analysis (CCA)
        self.n, self.p = np.shape(X)
        self.q = np.shape(Y)[1]
        # self.Xstd = pp.StandardScaler(with_std=False).fit_transform(X)
        # self.Ystd = pp.StandardScaler(with_std=False).fit_transform(Y)
        self.Xstd = pp.StandardScaler(with_std=True).fit_transform(X)
        self.Ystd = pp.StandardScaler(with_std=True).fit_transform(Y)
        # self.Xstd = utils.standardizare(X)
        # self.Ystd = utils.standardizare(Y)
        VX = np.cov(self.Xstd, rowvar=False)
        VY = np.cov(self.Ystd, rowvar=False)
        cov = np.cov(self.Xstd, self.Ystd, rowvar=False)
        VXY = cov[:self.p, self.p:]
        VYX = np.transpose(VXY)
        invVX = np.linalg.inv(VX)
        invVY = np.linalg.inv(VY)
        h1 = invVX @ VXY
        h2 = invVY @ VYX
        self.m = min(self.p, self.q)
        if self.p == self.m:
            h = h1 @ h2
            eigenvalues, eigenvectors = np.linalg.eig(h)
            k_inv = [k for k in reversed(np.argsort(eigenvalues))]
            self.r2 = eigenvalues[k_inv]
            a = eigenvectors[:, k_inv]
            self.r2 = eigenvalues
            a = eigenvectors
            self.r = np.sqrt(self.r2)
            b = (h2 @ a) @ np.diag(1 / self.r)
            z = self.Xstd @ a
            u = self.Ystd @ b
        else:
            h = h2 @ h1
            eigenvalues, eigenvectors = np.linalg.eig(h)
            k_inv = [k for k in reversed(np.argsort(eigenvalues))]
            self.r2 = eigenvalues[k_inv]
            b = eigenvectors[:, k_inv]
            self.r2 = eigenvalues
            b = eigenvectors
            self.r = np.sqrt(self.r2)
            a = (h1 @ b) @ np.diag(1 / self.r)
            z = self.Xstd @ a
            u = self.Ystd @ b
        self.z = pp.normalize(z, axis=0)
        self.u = pp.normalize(u, axis=0)


    def getCanonicalRoots(self):
        return self.r, self.r2, self.z, self.u


    def chi2BartlettWilks_test(self):
        r_inv = np.flipud(self.r)
        # r_inv = np.linalg.inv(self.r)
        print('R:', self.r)
        print('R inverse:', r_inv)
        dof = (self.p - np.arange(self.m)) * (self.q - np.arange(self.m))
        print('dof:', dof)
        l = np.flipud(np.cumprod(1 - (r_inv * r_inv)))
        # l = np.linalg.inv(np.cumprod(1 - (r_inv * r_inv)))
        print('lambda:', l)
        chi2Calc = (-self.n + 1 + (self.p + self.q + 1) / 2) * np.log(l)
        chi2Tab = 1 - sts.chi2.cdf(chi2Calc, dof)
        return chi2Calc, chi2Tab, dof