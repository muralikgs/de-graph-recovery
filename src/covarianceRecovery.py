import numpy as np 

from utils import *

# TODO implement a function that can choose between different covariance recovery methods
def getCovariance(Y, A, method='exact-sol', pen_coeff=0.5):
    Sig_Y = (1/Y.shape[0]) * Y.T @ Y
    if method == 'exact-sol':
        Cov_est = covarianceEstimate(Sig_Y, A, pen_coeff)
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


