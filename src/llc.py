import numpy as np

def replace_submatrix(mat, ind1, ind2, mat_replace):
  for i, index in enumerate(ind1):
    mat[index, ind2] = mat_replace[i, :]
  return mat

def get_gt_covariance(B, n_nodes, intervention_set, int_scale=1.0, noise_scale=0.5): 
    # B is the transpose of the weights matrix
    Cov_x = np.zeros((n_nodes, n_nodes))
    
    observed_set = np.setdiff1d(np.arange(n_nodes), intervention_set)

    mat_t = np.linalg.inv(np.eye(len(observed_set)) - B[observed_set, :][:, observed_set])
    cross_weights = B[observed_set, :][:, intervention_set]
    T =  mat_t @ cross_weights
    C_obs = mat_t @ ( cross_weights @ cross_weights.T + (noise_scale**2)*np.eye(len(observed_set)) ) @ mat_t.T

    Cov_x = replace_submatrix(Cov_x, intervention_set, intervention_set, (int_scale**2) * np.eye(len(intervention_set)))
    Cov_x = replace_submatrix(Cov_x, observed_set, intervention_set, T)
    Cov_x = replace_submatrix(Cov_x, intervention_set, observed_set, T.T)
    Cov_x = replace_submatrix(Cov_x, observed_set, observed_set, C_obs)

    return Cov_x

def get_coefficients(cov, i, u, intervention_set, observed_set):
    coefs = np.zeros(len(intervention_set) + len(observed_set)-1)
    
    get_index = lambda x: x if x < u else x-1
    for node in observed_set:
        if node != u:
            coefs[get_index(node)] = cov[i, node]
    
    coefs[get_index(i)] = 1
    return coefs

def parse_experiment(est_cov, intervention_set, T, t, curr_row=0, use_ground_truth_cov=False, B=None, int_scale=1.0, noise_scale=0.5):
    n_nodes = est_cov.shape[1]
    observed_set = np.setdiff1d(np.arange(n_nodes), intervention_set)

    # step 1 - Get the covariance matrix
    # Cov_x = (1/dataset.shape[0]) * dataset.T @ dataset

    st_row = curr_row
    #step 3 - construct T and t
    for int_node in intervention_set:
        for obs_node in observed_set:
            coefs = get_coefficients(est_cov, int_node, obs_node, intervention_set, observed_set)
            st_col = obs_node * (n_nodes - 1)
            T[st_row, st_col : st_col + n_nodes - 1] = coefs
            t[st_row] = est_cov[int_node, obs_node]
            st_row += 1
            
    return T, t, st_row

def compute_n_rows(n_nodes, intervention_sets):
    n_rows = 0
    for intervention_set in intervention_sets:
        n_rows += len(intervention_set) * (n_nodes - len(intervention_set))
    
    return n_rows

def predict_adj_llc(est_cov_list, intervention_sets, use_ground_truth_cov=False, B=None, int_scale=1.0, noise_scale=0.5):
    n_nodes = est_cov_list[0].shape[1]
    n_rows = compute_n_rows(n_nodes, intervention_sets)
    n_cols = n_nodes * (n_nodes - 1)

    T = np.zeros((n_rows, n_cols))
    t = np.zeros((n_rows, 1))
    st_row = 0

    i = 0
    for est_cov, intervention_set in zip(est_cov_list, intervention_sets):

        # if intervention_set[0] != None:
        # print("parsing experiment: {}".format(i))
        i += 1
        T, t, st_row = parse_experiment(est_cov, intervention_set, T, t, st_row, use_ground_truth_cov, B, int_scale, noise_scale)

    b_est = np.linalg.pinv(T) @ t
    B_est = np.zeros((n_nodes, n_nodes))
    for n in range(n_nodes):
        exc_n_set = np.setdiff1d(np.arange(n_nodes), n)
        B_est[exc_n_set, n] = b_est[n * (n_nodes-1) : (n + 1) * (n_nodes - 1)].squeeze()

    return T, t, B_est