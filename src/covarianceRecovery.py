import numpy as np 
import torch
from cvxpylayers.torch import CvxpyLayer

from utils import *

def covarianceEstimateTorch(A, Sig_Y, pen_coeff=0.5):

    Sigma_hat = cp.Parameter(Sig_Y.shape)
    m, n = A.shape

    xhat = cp.Variable((n,n), PSD=True)
    objective = cp.Minimize(cp.sum(cp.abs(xhat)))
    constraints = [cp.norm(A @ xhat @ A.T - Sigma_hat)**2 <= pen_coeff]

    prob = cp.Problem(objective, constraints)
    assert prob.is_dpp()

    cvxpylayer = CvxpyLayer(prob, parameters=[Sigma_hat], variables=[xhat])
    # A_tch = torch.tensor(A_bs, requires_grad=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Sig_tch = torch.tensor(Sig_Y, requires_grad=True, device=device)

    solution, = cvxpylayer(Sig_tch)

    Sig_est = (solution + solution.T)/2
    return Sig_est.detach().cpu().numpy()


# TODO implement a function that can choose between different covariance recovery methods
def getCovariance(Y, A, method='cvxpy', pen_coeff=0.5):
    Sig_Y = (1/Y.shape[0]) * Y.T @ Y
    if method == 'cxvpy':
        Cov_est = covarianceEstimate(Sig_Y, A, pen_coeff)
    elif method == 'cvxpylayers':
        Cov_est = covarianceEstimateTorch(A, Sig_Y, pen_coeff=pen_coeff)
    return Cov_est

def reorderCovariance(cov_r, indices):
    rev_indices = np.argsort(indices)
    
    col_ind_r, row_ind_r = np.meshgrid(rev_indices, rev_indices)
    cov = cov_r[row_ind_r, col_ind_r]

    return cov

def process_int_covariances(int_covariance_list, indices_list, n_int, n_h):
    cov_mat = np.zeros(int_covariance_list.shape)
    
    for int_cov_mat, indices in zip(int_covariance_list, indices_list):
        col_ind, row_ind = np.meshgrid(indices, indices)
        subblock_r, subblock_c = row_ind[:n_int, n_int:n_h], col_ind[:n_int, n_int:n_h]
        cov_mat[subblock_r, subblock_c] = int_cov_mat[subblock_r, subblock_c]
    
    cov_mat = cov_mat + cov_mat.T

    return cov_mat
    
# The following function recovers the covariance of Xs and rearranges it be compatible with the original ordering. 
def getIntCovariances(meas_list, indices_list, A, n_int, n_h, method='cvxpylayers'):
    covariance_list = list()
    for Y_list, indices in zip(meas_list, indices_list):
        int_covariance_list = list()
        for Y, indice in zip(Y_list, indices):
            covariance_est_r = getCovariance(Y, A, method=method)
            covariance_est = reorderCovariance(covariance_est_r, indice)
            int_covariance_list.append(covariance_est)

        covariance_list.append(
            process_int_covariances(int_covariance_list, indices, n_int, n_h)
        )

    return covariance_list
