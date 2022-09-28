import numpy as np
import cvxpy as cp

def getSensingMatrix(rho_vec, lambda_vec, n, A=1):
    dv = len(lambda_vec)
    dc = len(rho_vec)
    
    mTemp = np.round(n * np.arange(1, dv+1).reshape(1, dv) @ lambda_vec/ (np.arange(1, dc+1).reshape(1, dc) @ rho_vec)).squeeze()
    rowDegDist = np.round(mTemp * rho_vec)
    m = int(np.sum(rowDegDist))
    colDegDist = np.round(n * lambda_vec)
    
    excessN = colDegDist.sum() - n
    
    i = 1
    while excessN > 0:
        if colDegDist[i] > 0:
            colDegDist[i] -= 1
        i += 1
        excessN = colDegDist.sum() - n

    i = 1
    while excessN < 0:
        colDegDist[i] += 1
        i += 1
        excessN = colDegDist.sum() - n
                
    rowDegree = np.zeros(m)
    rows = list(range(m))
    for i in range(dc):
        if rowDegDist[i] == 0:
            continue
        
        sampledRows = np.random.choice(rows, int(rowDegDist[i]), replace=False)
        rowDegree[sampledRows] = i+1
        rows = list(set(rows).difference(set(sampledRows)))
    
    colDegree = np.zeros(n)
    cols = list(range(n))
    for i in range(dv):
        if colDegDist[i] == 0:
            continue
        
        sampledCols = np.random.choice(cols, int(colDegDist[i]), replace=False)
        colDegree[sampledCols] = i+1
        cols = list(set(cols).difference(set(sampledCols)))
    
    nEdges = colDegree.sum()
    print("Total Edges: {}".format(nEdges))
    excessEdges = rowDegree.sum() - nEdges
    print("Excess edges: {}".format(excessEdges))
    i = 0 
    while excessEdges > 0:
        rowDegree[m-1-i] -= 1
        i += 1
        excessEdges = rowDegree.sum() - nEdges
    
    i = 0
    while excessEdges < 0:
        rowDegree[m-1-i] += 1
        i += 1
        excessEdges = rowDegree.sum() - nEdges
    
    print("T Edges: {}".format(nEdges))
    print("R Edges: {}".format(rowDegree.sum()))
    print("Sensing Matrix Dimensions: {} X {}".format(m, n))
    
    rowDegDict = { r: deg for r, deg in enumerate(rowDegree) }
    availRows = lambda x: [k for k in x if x[k] > 0]    
    
    A_mat = np.zeros((m, n))
    for col in range(n):
        freeRows = availRows(rowDegDict)
        sampledRows = np.random.choice(freeRows, int(colDegree[col]), replace=False)
        A_mat[sampledRows, col] = 1
        
    A_mat = A_mat * (2 * (np.random.rand(m, n) > 0.5) - 1) * (1/A)**0.5
        
    print("Matrix Dimensions: {} X {}".format(m, n))
    return A_mat
    
def getSensingMatrixUP(lambda_vec_h, lambda_vec_l, dch, dcl, nh, nl, A=1):
    dvh = len(lambda_vec_h)
    dvl = len(lambda_vec_l)
    
    m = int(np.round((nh * np.arange(1, dvh+1).reshape(1, dvh) @ lambda_vec_h + nl * np.arange(1, dvl+1).reshape(1, dvl) @ lambda_vec_l)/(dch + dcl)))
    A_mat = np.zeros((m, nh+nl))
    
    # Construct the high priority connections part
    col_deg_high = np.round(nh * lambda_vec_h)
    
    excessN = col_deg_high.sum() - nh
    # print(excessN)
    i = 1
    while excessN > 0:
        if col_deg_high[i] > 0:
            col_deg_high[i] -= 1
        i += 1
        excessN = col_deg_high.sum() - nh
        
    i = 1
    while excessN < 0:
        col_deg_high[i] += 1
        i += 1
        excessN = col_deg_high.sum() - nh
        
    # print(col_deg_high)
    
    colDegree = np.zeros(nh)
    cols = list(range(nh))
    for i in range(dvh):
        if col_deg_high[i] == 0:
            continue
        
        sampledCols = np.random.choice(cols, int(col_deg_high[i]), replace=False)
        colDegree[sampledCols] = i+1
        cols = list((set(cols)).difference(set(sampledCols)))
    
    nEdges = m*dch
    excessEdges = colDegree.sum() - nEdges
    # print(excessEdges)
    i = 0
    while excessEdges > 0:
        if i == nh:
            i = 0
        colDegree[nh-1-i] -= 1
        i += 1
        excessEdges = colDegree.sum() - nEdges
    
    i = 0
    while excessEdges < 0:
        if i == nh:
            i = 0
        colDegree[nh-1-i] += 1
        i += 1
        excessEdges = colDegree.sum() - nEdges
        
    colDegDict = { c: deg for c, deg in enumerate(colDegree)}
    availCols = lambda x: [k for k in x if x[k] > 0]
    
    for row in range(m):
        freeCols = availCols(colDegDict)
        sampledCols = np.random.choice(freeCols, dch, replace=False)
        A_mat[row, sampledCols] = 1

        
    # Constructing the low priority part
    
    col_deg_low = np.round(nl * lambda_vec_l)
    
    excessN = col_deg_low.sum() - nl
    i = 1
    while excessN > 0:
        if col_deg_low[i] > 0:
            col_deg_low[i] -= 1
        i += 1
        excessN = col_deg_low.sum() - nl
        
    i = 1
    while excessN < 0:
        col_deg_low[i] += 1
        i += 1
        excessN = col_deg_low.sum() - nl
    
    colDegree = np.zeros(nl)
    cols = list(range(nl))
    for i in range(dvl):
        if col_deg_low[i] == 0:
            continue
        
        sampledCols = np.random.choice(cols, int(col_deg_low[i]), replace=False)
        colDegree[sampledCols] = i+1
        cols = list((set(cols)).difference(set(sampledCols)))
    
    nEdges = m*dcl
    excessEdges = colDegree.sum() - nEdges
    i = 0
    while excessEdges > 0:
        if i == nl:
            i = 0
        colDegree[nl-1-i] -= 1
        i += 1
        excessEdges = colDegree.sum() - nEdges
    
    i = 0
    while excessEdges < 0:
        if i == nl:
            i = 0
        colDegree[nl-1-i] += 1
        i += 1
        excessEdges = colDegree.sum() - nEdges
        
    colDegDict = { c: deg for c, deg in enumerate(colDegree)}
    availCols = lambda x: [k for k in x if x[k] > 0]
    
    for row in range(m):
        freeCols = availCols(colDegDict)
        sampledCols = np.random.choice(freeCols, dch, replace=False)
        modsampledCols = [col + nh for col in sampledCols]
        A_mat[row, modsampledCols] = 1
        
    A_mat = A_mat * (2 * (np.random.rand(m, nh + nl) > 0.5) - 1) * (1/A)**0.5
        
    return A_mat

def covarianceEstimate(Sigma_hat, A, pen_coeff=None, noise=True):

    p = A.shape[1]

    xhat = cp.Variable((p,p), PSD=True)
    objective = cp.Minimize(cp.sum(cp.abs(xhat)))
    if noise:
        constraints = [cp.norm(A @ xhat @ A.T - Sigma_hat)**2 <= pen_coeff]
    else:
        constraints = [A @ xhat @ A.T - Sigma_hat == 0]

    prob = cp.Problem(objective, constraints)
    results = prob.solve(solver=cp.SCS)

    Sigma_est = xhat.value

    # ind_mat = Sigma_est < Sigma_est.T
    # Sigma_est = ind_mat * Sigma_est + (1 - ind_mat) * Sigma_est.T

    Sigma_est = (Sigma_est + Sigma_est.T)/2

    return Sigma_est

def error_metrics(W_gt, W_est):

    TP = ((W_gt == W_est)*W_gt).sum()
    TN = ((W_gt == W_est)*(1-W_gt)).sum()

    FP = W_est.sum() - TP
    FN = (1 - W_est).sum() - TN

    accuracy = (TP + TN)/(TP + TN + FP + FN)
    precision = TP/(TP + FP)
    recall = TP/(TP + FN)
    f1 = 2*precision*recall/(precision + recall)
    
    return (TP, TN, FP, FN), (accuracy, precision, recall, f1)

def generateA(m, p, delta, M=5, prob=0.5, replace=False, type='bipartite'):
    
    if type == "bipartite":
        A = np.zeros((m, p))
        rows = list(range(m))
        
        for i in range(p):
            rand_rows = np.random.choice(rows, size=delta, replace=replace)
            for r in rand_rows:
                A[r, i] += 1

    elif type == "gaussian":
        scale = (1/m)**0.5
        A = np.random.normal(0, scale=scale, size=(m, p))

    elif type == "integer":
        A = np.random.binomial(n=1, p=prob, size=(m, p))
        int_mat = np.random.randint(low=1, high=M+1, size=(m, p))
        A = A * int_mat
            
    return A

def parents2Adjacency(parents):
	nodes = parents.keys()
	
	W = np.zeros((len(nodes), len(nodes)))
	for node in nodes:
		W[parents[node], node] = 1

	return W

def SHD(W_1, W_2):
	# Reverse edges
	R = (W_2.T == W_1)*W_1

	W_1_m = W_1 - R 
	W_2_m = W_2 - R.T 
	
	# Extra edges
	E = W_2_m > W_1_m

	# Missing edges
	M = W_2_m < W_1_m

	return R.sum() + E.sum() + M.sum() 

def constructSensingMatUP(lambda_vec_h, lambda_vec_l, dch, dcl, nh, nl, A=1):
    dvh = len(lambda_vec_h)
    dvl = len(lambda_vec_l)
    
    m = int(np.round((nh * np.arange(1, dvh+1).reshape(1, dvh) @ lambda_vec_h + nl * np.arange(1, dvl+1).reshape(1, dvl) @ lambda_vec_l)/(dch + dcl)))
    A_mat = np.zeros((m, nh+nl))
    
    # Construct the high priority connections part
    col_deg_high = np.round(nh * lambda_vec_h)
    
    excessN = col_deg_high.sum() - nh
    
    i = 1
    while excessN > 0:
        if col_deg_high[i] > 0:
            col_deg_high[i] -= 1
        i += 1
        excessN = col_deg_high.sum() - nh
        
    i = 1
    while excessN < 0:
        col_deg_high[i] += 1
        i += 1
        excessN = col_deg_high.sum() - nh
    
    colDegree = np.zeros(nh)
    cols = list(range(nh))
    for i in range(dvh):
        if col_deg_high[i] == 0:
            continue
        
        sampledCols = np.random.choice(cols, int(col_deg_high[i]), replace=False)
        colDegree[sampledCols] = i+1
        cols = list((set(cols)).difference(set(sampledCols)))
    
    nEdges = m*dch
    excessEdges = colDegree.sum() - nEdges
    print(excessEdges)
    i = 0
    while excessEdges > 0:
        colDegree[nh-1-i] -= 1
        i += 1
        excessEdges = colDegree.sum() - nEdges
    
    i = 0
    while excessEdges < 0:
        colDegree[nh-1-i] += 1
        i += 1
        excessEdges = colDegree.sum() - nEdges
        
    colDegDict = { c: deg for c, deg in enumerate(colDegree)}
    availCols = lambda x: [k for k in x if x[k] > 0]
    
    for row in range(m):
        freeCols = availCols(colDegDict)
        sampledCols = np.random.choice(freeCols, dch, replace=False)
        A_mat[row, sampledCols] = 1

        
    # Constructing the low priority part
    
    col_deg_low = np.round(nl * lambda_vec_l)
    
    excessN = col_deg_low.sum() - nl
    i = 1
    while excessN > 0:
        if col_deg_low[i] > 0:
            col_deg_low[i] -= 1
        i += 1
        excessN = col_deg_low.sum() - nl
        
    i = 1
    while excessN < 0:
        col_deg_low[i] += 1
        i += 1
        excessN = col_deg_low.sum() - nl
    
    colDegree = np.zeros(nl)
    cols = list(range(nl))
    for i in range(dvl):
        if col_deg_low[i] == 0:
            continue
        
        sampledCols = np.random.choice(cols, int(col_deg_low[i]), replace=False)
        colDegree[sampledCols] = i+1
        cols = list((set(cols)).difference(set(sampledCols)))
    
    nEdges = m*dcl
    excessEdges = colDegree.sum() - nEdges
    i = 0
    while excessEdges > 0:
        colDegree[nl-1-i] -= 1
        i += 1
        excessEdges = colDegree.sum() - nEdges
    
    i = 0
    while excessEdges < 0:
        colDegree[nl-1-i] += 1
        i += 1
        excessEdges = colDegree.sum() - nEdges
        
    colDegDict = { c: deg for c, deg in enumerate(colDegree)}
    availCols = lambda x: [k for k in x if x[k] > 0]
    
    for row in range(m):
        freeCols = availCols(colDegDict)
        sampledCols = np.random.choice(freeCols, dch, replace=False)
        modsampledCols = [col + nh for col in sampledCols]
        A_mat[row, modsampledCols] = 1
        
    A_mat = A_mat * (2 * (np.random.rand(m, nh + nl) > 0.5) - 1) * (1/A)**0.5
        
    return A_mat
