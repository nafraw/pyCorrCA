import numpy as np
from scipy import linalg as sp_linalg
from scipy import diag as sp_diag

try:
    import matlab.engine
    matlab_imported = True
except ModuleNotFoundError:
    matlab_imported = False
    print(f"MATLAB Engine cannot be imported.\n" +
          f"You still can use the CorrCA class, but you cannot run verification with the MATLAB version\n" +
          f"If you wish to run the verification. You will need to work on solving the MATLAB path/installation issue\n" +
          f"Potential solutions can be found at:\n" +
          f"https://www.mathworks.com/matlabcentral/answers/362824-no-module-named-matlab-engine-matlab-is-not-a-package\n" +
          f"If you don't have internet, try to locate MATLABroot/extern/engines/python and run python setup.py install with admin right"
         )


# Revised from https://github.com/renzocom/CorrCA
# What are different:
    # There is a real class now and should support scikit-learn format (fit, transform, and fit_transform)
    # Verify whether the implementation is consistent with MATLAB. Previous version is different, mainly due to eigen value decomposition. See the NOTE below for explanation.
    # The current implementation should be consistent. However, the directions (signs) of eigen vectors and numerical precisions may still not match.
    # Some functions were removed but one should not find it difficult to add back.

## NOTE:
# MATLAB uses Cholesky algorithm for eigen value decomposition while Python uses QZ (for non-symmetric matrix) if calling eig
# Calling eigh in Python should lead to equivalent eig in MATLAB for a symmetric matrix
# ref: https://stackoverflow.com/questions/8765310/scipy-linalg-eig-return-complex-eigenvalues-for-covariance-matrix

# since scipy.linalg.issymmetry does not work...
def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

class CorrCA:
    def __init__(self):
        pass

    def reverse_dim(self, X):
        # call this function when X is time x dimensions x subjects (MATLAB version)
        return np.transpose(X, axes=[2, 1, 0])

    def fit(self, X, version=2, gamma=0, nComponent=None, reverse=True):
        # X can be in either format below
        # X: time x dimensions x subjects (MATLAB version, so reverse is True by default)
        # X: subjects x dimensions x time (Python version, so reverse should be set as False)
        if X.ndim == 2: X = X[:, :, np.newaxis]
        if reverse: X = self.reverse_dim(X)
        self.reverse = reverse
        self._fit(X, version=version, gamma=gamma, nComponent=nComponent)

    def transform(self, X, reverse=None):
        # project from signal space to component space
        # X: signal space (time x dimensions x subjects)
        # Y: correlated component space
        if reverse == None: reverse = self.reverse
        if reverse: X = self.reverse_dim(X)
        assert(X.ndim==3)
        N, _, T = X.shape
        Y = np.zeros((N, self.nComponent, T))
        for n in range(N):
            Y[n, ...] = self.W.T.dot(X[n, ...])
        return Y

    def reconstruct(self, Y):
        # project from correlated component space to signal space
        # Y: correlated component
        N, _, T = Y.shape
        X = np.zeros((N, self.D, T))
        for n in range(N):
            X[n, ...] = self.A.dot(Y[n, ...])
        return X

    def fit_transform(self, X, version=2, gamma=0, nComponent=None, reverse=True):
        self.fit(X, version=version, gamma=gamma, nComponent=nComponent, reverse=reverse)
        Y = self.transform(X, reverse=reverse)
        return Y

    def regInv(self, R, k):
        '''PCA regularized inverse of square symmetric positive definite matrix R.'''
        U, S, Vh = np.linalg.svd(R)
        invR = U[:, :k].dot(sp_diag(1 / S[:k])).dot(Vh[:k, :])
        return invR

    def get_RW_Rt_Rb(self, X):
        N, D, T = X.shape
        if self.version == 1:
            Xcat = X.reshape((N * D, T)) # T x (D + N) note: dimensions vary first, then subjects
            Rkl = np.cov(Xcat).reshape((N, D, N, D)).swapaxes(1, 2)
            Rw = Rkl[range(N), range(N), ...].sum(axis=0) # Sum within subject covariances
            Rt = Rkl.reshape(N*N, D, D).sum(axis=0)
            Rb = (Rt - Rw) / (N-1)
        elif self.version == 2:
            Rw = sum(np.cov(X[n,...]) for n in range(N))
            Rt = N**2 * np.cov(X.mean(axis=0))
            Rb = (Rt - Rw) / (N-1)
        elif self.version == 3:
            Rw = sum(np.cov(X[n,...]) for n in range(N))
            Rt = N**2 * np.cov(X.mean(axis=0))
            Rb = Rt
        else:
            assert(f'Error: version={self.version} while it must be either 1, 2, or 3')
        return Rw, Rt, Rb

    def get_ISC(self, X):
        Rw, _, Rb = self.get_RW_Rt_Rb(X)
        ISC = np.diagonal(self.W.T.dot(Rb).dot(self.W)) / np.diag(self.W.T.dot(Rw).dot(self.W))
        return np.real(ISC)

    def get_forward_model(self):
        # forward model: component to signal (time)
        if self.k==self.D:
            A = self.Rw.dot(self.W).dot(sp_linalg.inv(self.W.T.dot(self.Rw).dot(self.W)))
        else:
            A = self.Rw.dot(self.W).dot(np.diag(1 / np.diag(self.W.T.dot(self.Rw).dot(self.W))))
        self.A = A

    def _fit(self, X, version=2, gamma=0, nComponent=None):
        assert(X.ndim==3)
        self.version = version
        self.gamma = gamma
        k = nComponent

        N, D, T = X.shape # time x dimensions x subjects
        self.D = D
        if k is None:
            k = D

        Rw, Rt, Rb = self.get_RW_Rt_Rb(X)


        k = min(k, np.linalg.matrix_rank(Rw)) # handle rank deficient data.
        if k < D:
            invR = self.regInv(Rw, k)
            # since it is not a square matrix, definitely not symmetric, directly use eig()
            ISC, W = sp_linalg.eig(invR.dot(Rb)) # Eigen vectors are not unique, one cannot make sure if MATLAB and Python versions match as the eig() are implemented differently

            _, W, idx = sort_eigen(np.abs(ISC), W) # sort with absolute value, because a too small eigen value is likely numeric unstable
            ISC, W = ISC[idx[:k]], W[:, :k] # only takes k components
            # sort again using non-absolute values
            ISC, W, _ = sort_eigen(ISC, W)

        else:
            Rw_reg = (1-gamma) * Rw + gamma * Rw.diagonal().mean() * np.identity(D)
            ISC, W = sp_linalg.eigh(Rb, Rw_reg) # Eigen vectors are not unique, one cannot make sure if MATLAB and Python versions match as the eig() are implemented differently
            ISC, W, _ = sort_eigen(ISC, W)
            W = normalize_col(W) # normailze needed when calling eigh, but not eig()

        self.W = np.real(W)
        self.ISC = self.get_ISC(X)
        self.nComponent = k
        self.Rw = Rw
        self.Rb = Rb
        self.Rt = Rt


def sort_eigen(eigv, V):
    idx = eigv.argsort()[::-1]
    eigv = eigv[idx]
    V = V[:,idx]
    return eigv, V, idx

def normalize_col(W):
    # print('W', W)
    # print('s', np.sqrt(sum(np.square(W))))
    W=W/np.sqrt(sum(np.square(W)))
    return W


### --- For verification purpose --- ###
def corrca_matlab(X, version=2, W=None, gamma=0, k=None):
    assert(X.ndim==3)
    X = np.transpose(X, axes=[2, 1, 0]) # before tranpose: time x dimensions x subjects, matches MATLAB version
    N, D, T = X.shape # time x dimensions x subjects
    if k is None:
        k = D

    if version == 1:
        Xcat = X.reshape((N * D, T)) # T x (D + N) note: dimensions vary first, then subjects
        Rkl = np.cov(Xcat).reshape((N, D, N, D)).swapaxes(1, 2)
        Rw = Rkl[range(N), range(N), ...].sum(axis=0) # Sum within subject covariances
        Rt = Rkl.reshape(N*N, D, D).sum(axis=0)
        Rb = (Rt - Rw) / (N-1)
    elif version == 2:
        Rw = sum(np.cov(X[n,...]) for n in range(N))
        Rt = N**2 * np.cov(X.mean(axis=0))
        Rb = (Rt - Rw) / (N-1)
    elif version == 3:
        Rw = sum(np.cov(X[n,...]) for n in range(N))
        Rt = N**2 * np.cov(X.mean(axis=0))
        Rb = Rt
    else:
        assert(f'Error: version={version} while it must be either 1, 2, or 3')

    if W is None:
        k = min(k, np.linalg.matrix_rank(Rw)) # handle rank deficient data.
        if k < D:
            def regInv(R, k):
                '''PCA regularized inverse of square symmetric positive definite matrix R.'''
                U, S, Vh = np.linalg.svd(R)
                invR = U[:, :k].dot(sp_diag(1 / S[:k])).dot(Vh[:k, :])
                return invR

            invR = regInv(Rw, k)
            # since it is not a square matrix, definitely not symmetric, directly use eig()
            ISC, W = sp_linalg.eig(invR.dot(Rb)) # Eigen vectors are not unique, one cannot make sure if MATLAB and Python versions match as the eig() are implemented differently

            _, W, idx = sort_eigen(np.abs(ISC), W) # sort with absolute value, because a too small eigen value is likely numeric unstable
            ISC, W = ISC[idx[:k]], W[:, :k] # only takes k components
            # sort again using non-absolute values
            ISC, W, _ = sort_eigen(ISC, W)
        else:
            Rw_reg = (1-gamma) * Rw + gamma * Rw.diagonal().mean() * np.identity(D)
            ISC, W = sp_linalg.eigh(Rb, Rw_reg) # Eigen vectors are not unique, one cannot make sure if MATLAB and Python versions match as the eig() are implemented differently
            ISC, W, _ = sort_eigen(ISC, W)
            W = normalize_col(W) # eigh() needs normalization, eig() does not


    ISC = np.diagonal(W.T.dot(Rb).dot(W)) / np.diag(W.T.dot(Rw).dot(W))
    ISC, W = np.real(ISC), np.real(W)

    Y = np.zeros((N, k, T))
    for n in range(N):
        Y[n, ...] = W.T.dot(X[n, ...])

    if k==D:
        A = Rw.dot(W).dot(sp_linalg.inv(W.T.dot(Rw).dot(W)))
    else:
        A = Rw.dot(W).dot(np.diag(1 / np.diag(W.T.dot(Rw).dot(W))))

    #TODO: p is not implemented yet...

    return W, ISC, Y, A

def compare_matlab_python(data_np=np.random.rand(3,6,20), version=2, gamma=0, path_MATLAB_lib='./'):
    if not matlab_imported:
        print('MATLAB engine not imported, can\'t run the function')
        return None
    version = 2
    gamma = 0
    eng = matlab.engine.start_matlab()
    eng.cd(path_MATLAB_lib, nargout=0)

    data = matlab.double(data_np.tolist())
    W_, ISC_, Y_, A_ = eng.corrca(data, {'version': version, 'gamma': gamma}, nargout=4) # MATLAB results
    W_ = np.asarray(W_)
    ISC_ = np.asarray(ISC_)
    W, ISC, Y, A = corrca_matlab(data_np, version=version, gamma=gamma) # Direct implementation from MATLAB code
    CA = CorrCA() # Implementation made in class
    Y_class = CA.fit_transform(data_np, version=version, gamma=gamma, nComponent=None, reverse=True)

    # Uncomment if one would like to see detailed difference
    # print('Difference of eigen vectors (they can be very different as eigen vectors are not a unique set):')
    # print(W-np.asarray(W_))
    # print(CA.W-np.asarray(W_))

    # revert sign, eigen vectors can point to the other side due to algorithm implementations
    different_sign_class = 0
    different_sign_function = 0
    for i in range(0, W.shape[0]):
        if np.sign(W[0, i]) != np.sign(W_)[0, i]:
            W[:, i] *= - 1
            different_sign_function+=1
        if np.sign(CA.W[0, i]) != np.sign(W_)[0, i]:
            CA.W[:, i] *= - 1
            different_sign_class+=1
    print(f"------ Difference between the implemented CorrCA function and MATLAB ------")
    print(f"There is/are {different_sign_function} eigenvectors returned with different sign(s) from MATLAB")
    print(f'Sum of absolute difference of eigenvectors: {np.sum(np.abs((W-W_)))}')
    print(f'Max of absolute difference of eigenvectors: {np.max(np.abs((W-W_)))}')
    print(f'Sum of absolute difference of eigenvalues: {np.sum(np.abs((ISC-ISC_.T)))}')
    print(f'Max of absolute difference of eigenvalues: {np.max(np.abs((ISC-ISC_.T)))}')
    print(f"------ Difference between the implemented CorrCA class and MATLAB ------")
    print(f"There is/are {different_sign_class} eigenvectors returned with different sign(s) from MATLAB")
    print(f'Sum of absolute difference of eigenvectors: {np.sum(np.abs((CA.W-W_)))}')
    print(f'Max of absolute difference of eigenvectors: {np.max(np.abs((CA.W-W_)))}')
    print(f'Sum of absolute difference of eigenvalues: {np.sum(np.abs((CA.ISC-ISC_.T)))}')
    print(f'Max of absolute difference of eigenvalues: {np.max(np.abs((CA.ISC-ISC_.T)))}')

if __name__ == '__main__':
    from scipy import io
    import pickle
    time, ch, subject = 1000, 32, 20
    data_np = np.random.rand(time, ch, subject)
    ## Use case
    CA = CorrCA()
    reconstructed = CA.fit_transform(data_np, version=2, gamma=0.5, nComponent=None, reverse=True)
    CA.W # weight matrix
    CA.ISC # ISC for each component
    
    ### Verification between Python and MATLAB
    ## save generated random data if one needs to further verify on MATLAB side or do a post check on Python
    io.savemat('CorrCA_testData.mat', {'data': data_np})
    with open('CorrCA_testData.pkl', 'wb') as f:  # wb: overwrites any existing file.
        pickle.dump(data_np, f, pickle.HIGHEST_PROTOCOL)
    path_MATLAB_lib = './MATLAB'
    ## Instructions
    print('To verify with the MATLAB version. You will need to have MATLAB installed')
    print(f'Plus, the path_MATLAB_lib variable above, pointing to {path_MATLAB_lib}, should be set to the folder with corrca.m')
    print('One can download the MATLAB code from: https://www.parralab.org/corrca/')
    ## Verification
    print('------Comparing version 1------')
    compare_matlab_python(data_np=data_np, version=1, path_MATLAB_lib=path_MATLAB_lib)
    print('------Comparing version 2------')
    compare_matlab_python(data_np=data_np, version=2, gamma=0.5, path_MATLAB_lib=path_MATLAB_lib)
    print('------Comparing version 3------')
    compare_matlab_python(data_np=data_np, version=3, path_MATLAB_lib=path_MATLAB_lib)