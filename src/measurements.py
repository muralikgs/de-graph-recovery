import numpy as np 
import math

def cyclic_perm(a):
    n = len(a)
    b = [[a[i - j] for i in range(n)] for j in range(n)]
    return b

def generateIndicesSets(intervention_set, n, nh):
    # Generates indices sets used to reorder X
    #
    # A subset of the purely observed nodes are added to the high priority part to 
    # ensure recovery of the cross covariance between intervened nodes and the subset of 
    # observed nodes with high accuracy. 
    # The subsets are chosen such that each observed node is present in the high priority region
    # in atleast one index set. 

    vertex_set = np.arange(n)
    observed_set = np.setdiff1d(vertex_set, intervention_set)

    n_hp_obs_nodes = nh - len(intervention_set)
    n_subsets = math.ceil(len(observed_set) / n_hp_obs_nodes)
    partitioned_obs_sets = list()
    for subset_id in range(n_subsets):
        if (subset_id + 1) * n_hp_obs_nodes > len(observed_set):
            partitioned_obs_sets.append(
                observed_set[subset_id*n_hp_obs_nodes:].tolist()
            )
        else:
            partitioned_obs_sets.append(
                observed_set[subset_id*n_hp_obs_nodes : (subset_id + 1)*n_hp_obs_nodes].tolist()
            )

    subset_order_list = cyclic_perm(list(range(n_subsets)))
    indices_list = list()
    for i in range(n_subsets):
        obs_set = np.array([node for t in subset_order_list[i] for node in partitioned_obs_sets[t]])
        indices_list.append(
            np.concatenate((intervention_set, obs_set))
        )

    return indices_list

def makeIntMeasurements(A, X, intervention_set, nh):
    # vertex_set = np.arange(X.shape[1])
    # observed_set = np.setdiff1d(vertex_set, intervention_set)

    indices_list = generateIndicesSets(intervention_set, X.shape[1], nh)

    Y_cov_list = list()
    for indices in indices_list:
        X_r = X[:, indices]
        Y = X_r @ A.T
        cov = (1/Y.shape[0]) * Y.T @ Y
        Y_cov_list.append(cov)

    return Y_cov_list, indices_list

def makeMeasurements(A, dataset_x, intervention_sets, nh):
    Y_cov_int_list = list()
    indices_list = list()

    for X, intervention_set in zip(dataset_x, intervention_sets):
        Y_cov_list, indices = makeIntMeasurements(A, X, np.array(intervention_set), nh)
        Y_cov_int_list.append(Y_cov_list)
        indices_list.append(indices)
    
    return Y_cov_int_list, indices_list
