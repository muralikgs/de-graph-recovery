import numpy as np 

def makeIntMeasurements(A, X, intervention_set):
    vertex_set = np.arange(X.shape[1])
    observed_set = np.setdiff1d(vertex_set, intervention_set)

    indices = np.concatenate((intervention_set, observed_set))
    X_r = X[:, indices]
    
    Y = X_r @ A.T

    return Y

def makeMeasurements(A, dataset_x, intervention_sets):
    meas_list = list()

    for X, intervention_set in zip(dataset_x, intervention_sets):
        Y = makeIntMeasurements(A, X, np.array(intervention_set))
        meas_list.append(Y)
    
    return meas_list
