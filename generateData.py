import numpy as np 
import networkx.generators.random_graphs as rg
import networkx as nx 
from tqdm import tqdm
import igraph as ig 
from scipy.linalg import expm

# generates synthetic graphs sampled from Erdos-Renyi model

def h(W):
    
    h = np.trace(expm(W*W)) - W.shape[0]
    return h

def isAcyclic(W):
    if h(W) == 0:
        return True
    else:
        return False

def sampleWeights(W, weight_low=0.5, weight_high=2.0):
    
	assert weight_low <= weight_high

	# s1 = np.random.uniform(low=weight_low, high=weight_high, size=W.shape)
	# s2 = np.random.uniform(low=-1*weight_high, high=-1*weight_low, size=W.shape)

	s1 = (1/2) * np.ones(W.shape)
	s2 = (-1/2) * np.ones(W.shape)

	# The matrices are generated to randomly sample 
	# elements from the two weights matrices defined above
	m1 = np.random.randint(low=2, size=W.shape) # m1_{ij} \in {0,1}
	m2 = np.ones(W.shape) - m1 

	return W*(s1*m1 + s2*m2)


def generateW(d, p, M, max_iterations, weight_low, weight_high):

	# w_matrix_list = []
	# top_order_list = []

	# i = 0
	# count = 0

	# while i < max_iterations and count < M:
	# 	gp = rg.erdos_renyi_graph(n = d, p = p, directed=True)
	# 	W = nx.to_numpy_array(gp)
	# 	if nx.is_directed_acyclic_graph(gp):
	# 		W = sampleWeights(W, weight_low, weight_high)
	# 		w_matrix_list.append(W)
	# 		top_order_list.append(list(nx.topological_sort(gp)))
	# 		count += 1
	# 	i += 1
	# 	print("Progress: {}/{} (count = {}/{})".format(i, max_iterations, count, M), end="\r", flush=True)
	# print()
	# return w_matrix_list, top_order_list

	def _random_permutation(M):
		P = np.random.permutation(np.eye(M.shape[0]))
		return P.T @ M @ P

	def _random_acyclic_orientation(B_und):
		return np.tril(_random_permutation(B_und), k=-1)

	def _graph_to_adjmat(G):
		return np.array(G.get_adjacency().data)

	w_matrix_list = list()

	for i in tqdm(range(M)):
		gp = ig.Graph.Erdos_Renyi(n=d, m=d)
		A_und = _graph_to_adjmat(gp)
		W = _random_acyclic_orientation(A_und)

		assert isAcyclic(W)

		W = sampleWeights(W)

		# assert isAcyclic(W), "{}".format(h(W))

		w_matrix_list.append(W.T)

	return w_matrix_list


def sampleBN(W, Z):
    
    d = W.shape[0]
    # x = np.zeros((d,))
    
    # for i in range(d):
    #     x[i] = np.dot(W[:,i], x) + n[i]
        
    return (np.linalg.inv(np.eye(d) - W.T) @ Z.T).T
    

def generateSynthData(d_list, M, N,
					  p = 0.11,
					  dim_percent=0.5,
					  max_iter_factor=10,
					  weight_low=0.5, 
					  weight_high=2.0, 
					  noise_sc=1, 
					  m_noise_sc=1.0,
					  N_ref = 500):
	'''
	Args:
		d_list 			: list containing the number of edges to be considered for the experiments
		M 	   			: Number of graphs for each d
		N      			: Number of samples per graph
		p 	   			: Edge probability in the Erdos-Renyi model
		dim_percent		: ratio of dim(Y) over dim(X)
		max_iter_factor : Maximum number of iterations as a factor of M before termination of graph generation process
		weight_low		: Smallest value |W_ij| can take
		weight_high		: Largest value |W_ij| can take
		noise_sc 		: Sampling noise std
		m_noise_sc		: Measurement noise std
	'''

	max_iterations = M*max_iter_factor

	synthetic_data = dict()

	for d in d_list:
		print("Number of Nodes: {}".format(d))
		data = dict()

		# Generating the matrices
		print("Generating adj. matrices")
		w_matrix_list = generateW(d, p, M, max_iterations, weight_low, weight_high)
		data["W"] = w_matrix_list

		# Sampling direct observations X from the BN
		data["X"] = list()

		print("Sampling direct observations")
		for i in tqdm(range(M)):
			noise = np.random.normal(loc=0, scale=noise_sc, size=(N, d))
			X = sampleBN(w_matrix_list[i], noise)
			data["X"].append(X)

		# Constructing the measurement matrix
		print("Constructing the measurement matrix")
		print("changed scale")
		Y_dim = int(dim_percent*d)
		scale = (1/Y_dim)**0.5
		sensing_matrix_A = np.random.normal(0, scale, size=(Y_dim, d))
		data["A"] = sensing_matrix_A

		# Generating indirect measurements
		print("Generating the indirect measurements")
		data["Y"] = list()
		for i in tqdm(range(M)):
			X_ref = data["X"][i][:N_ref]
			std_est = X_ref.var(axis=0)**0.5
			
			# A_gen = sensing_matrix_A * (1/std_est)

			# measure_noise = np.random.normal(0,m_noise_sc, size=(N, Y_dim))
			indirect_measurements = sensing_matrix_A.dot(data["X"][i].T).T # + measure_noise
			data["Y"].append(indirect_measurements)

		synthetic_data[d] = data


	return synthetic_data

if __name__ == '__main__':
	
	synthetic_data = generateSynthData(d_list=[30],
									   M=1,
									   N=1000)








	