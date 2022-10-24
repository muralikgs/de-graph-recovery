import numpy as np 
import os 
import argparse 
import scipy.io as sio

from src.measurements import makeIntMeasurements
from src.data.datagen import Dataset 
from utils import *

def getGTCovariance(W, intervention_set):
    n = W.shape[0]
    
    observed_set = np.setdiff1d(np.arange(n), intervention_set)
    U = np.zeros_like(W)
    U[observed_set, observed_set] = 1

    Cov_x = np.linalg.inv(np.eye(n) - U @ W.T) @ np.linalg.inv(np.eye(n) - U @ W.T).T 
    return Cov_x

def generateAndSave(args):

    data_gen = Dataset(n_nodes=args.nodes, 
                        expected_density=args.exp_dens,
                        n_samples=args.samples,
                        n_experiments=args.nodes,
                        min_targets=args.int_size,
                        max_targets=args.int_size,
                        mode='block-node')
    
    datasets = data_gen.generate()
    intervention_sets = data_gen.targets

    if not os.path.exists(args.dop):
        os.makedirs(args.dop)
    
    np.save(os.path.join(args.dop, "intervention_sets.npy"), intervention_sets)
    np.save(os.path.join(args.dop, "weights.npy"), data_gen.gen_model.weights)

    lambda_h = sio.loadmat(os.path.join(args.deg_pol_path, "lambda_h_2.mat"))['lambda_h']
    lambda_l = sio.loadmat(os.path.join(args.deg_pol_path, "lambda_l_2.mat"))["lambda_l"]

    A = getSensingMatrixUP(lambda_h, lambda_l, dch=10, dcl=10, nh=25, nl=75, A=1.0)
    A_bs = generateA(A.shape[0], A.shape[1], delta=4)

    np.save(os.path.join(args.dop, "sensing_mat_de.npy"), A)
    np.save(os.path.join(args.dop, "sensing_mat_bs.npy"), A_bs)

    gt_covariance_path = os.path.join(args.dop, "gt_covariance")
    if not os.path.exists(gt_covariance_path):
        os.makedirs(gt_covariance_path)
    for i, intervention_set in enumerate(intervention_sets):
        Cov_x = getGTCovariance(data_gen.gen_model.weights, intervention_set)
        np.save(os.path.join(gt_covariance_path, "cov_x_{}.npy".format(i)), Cov_x)
    
        intervention_path = os.path.join(args.dop, "intervention_{}".format(i))
        if not os.path.exists(intervention_path):
            os.makedirs(intervention_path)

        Y_cov_list, indices_list = makeIntMeasurements(A, datasets[0], intervention_set, args.nh)
        Y_cov_list_bs, _ = makeIntMeasurements(A_bs, datasets[0], intervention_set, args.nh)

        for t, Y_cov in enumerate(Y_cov_list):
            np.save(os.path.join(intervention_path, "cov_y_{}.npy".format(t)), Y_cov)
            np.save(os.path.join(intervention_path, "cov_y_bs_{}.npy".format(t)), Y_cov_list_bs[t])
            np.save(os.path.join(intervention_path, "indices_{}.npy".format(t)), indices_list[t])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--nodes', type=int, default=100)
    parser.add_argument('--exp-dens', type=int, default=1)
    parser.add_argument('--samples', type=int, default=5000)
    parser.add_argument('--dop', type=str)
    parser.add_argument('--int-size', type=int, default=5)
    parser.add_argument('--nh', type=int, default=25)
    parser.add_argument('--deg-pol-path', type=str, default='./')
    
    args = parser.parse_args()

    generateAndSave(args)

