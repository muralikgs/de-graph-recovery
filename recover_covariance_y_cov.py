import numpy as np 
import torch
from cvxpylayers.torch import CvxpyLayer
import argparse
import os 

from src.covarianceRecovery import covarianceEstimateTorch

def run_experiment(args):
    if not args.bs:
        A = np.load(os.path.join(args.dip, "sensing_matrix_de.npy"))
    else:
        A = np.load(os.path.join(args.dip, "sensing_matrix_bs.npy"))
    
    if not os.path.exists(args.dop):
        os.makedirs(args.dop)

    for i in range(args.buff):
        print("Processing: {}/{}".format(i, args.buff))
        if args.bs:
            inter_path = os.path.join(os.path.join(args.dip, "bs_measurements"), "inter_{}".format(args.st_int+i))
        else:
            inter_path = os.path.join(os.path.join(args.dip, "de_measurements"), "inter_{}".format(args.st_int+i))
        
        inter_out_path = os.path.join(args.dop, "inter_{}".format(args.st_int+i))
        if not os.path.exists(inter_out_path):
            os.makedirs(inter_out_path)
        
        for j in range(5):
            Y_cov = np.load(os.path.join(inter_path, "Y_cov_{}.npy".format(j)))
            Cov_X_rec = covarianceEstimateTorch(A, Y_cov, pen_coeff=0.5)
            np.save(os.path.join(inter_out_path, "X_cov_rec_{}.npy".format(j)), Cov_X_rec)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-st-int", type=int, default=0)
    parser.add_argument("-buff", type=int, default=5)
    parser.add_argument("-dop", type=str)
    parser.add_argument("-dip", type=str)
    parser.add_argument("--bs", action='store_true', default=False)
    # parser.add_argument("-nh", type=int, default=25)

    args = parser.parse_args()
    run_experiment(args)
