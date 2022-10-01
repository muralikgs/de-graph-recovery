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

def reorderCovariance(cov_r, intervention_set):
    vertex_set = np.arange(cov_r.shape[0])
    observed_set = np.setdiff1d(vertex_set, intervention_set)

    indices = np.concatenate((intervention_set, observed_set))
    rev_indices = np.argsort(indices)
    
    col_ind_r, row_ind_r = np.meshgrid(rev_indices, rev_indices)
    cov = cov_r[row_ind_r, col_ind_r]

    return cov


# The following function recovers the covariance of Xs and rearranges it be compatible with the original ordering. 
def getIntCovariances(meas_list, A, intervention_sets):
    covariance_list = list()
    for Y, intervention_set in zip(meas_list, intervention_sets):
        covariance_est_r = getCovariance(Y, A, method='exact_sol')
        covariance_est = reorderCovariance(covariance_est_r, intervention_set)
        covariance_list.append(covariance_est)


