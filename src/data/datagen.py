import numpy as np 
import networkx as nx

def make_non_cotractive(weights):
    s = np.linalg.svd(weights, compute_uv=False)
    scale = 1.0
    if s[0] <= 1.0:
        scale = 2/s[0]
    
    return scale * weights 

def make_contractive(weights):
    s = np.linalg.svd(weights, compute_uv=False)
    scale=1.1
    if s[0] >= 1.0:
        scale = 1.1 * s[0]
    
    return weights/scale

class DirectedGraphGenerator:
    """
    -------------------------------------------------------------------
    Create the structure of a Directed (potentially cyclic) graph
    -------------------------------------------------------------------
    Args:
    nodes (int)            : Number of nodes in the graph.
    expected_density (int) : Expected number of edges per node. 
    """

    def __init__ (self, nodes=30, expected_density=3, enforce_dag=False):
        self.nodes = nodes
        self.expected_density = expected_density  
        self.adjacency_matrix = np.zeros((self.nodes, self.nodes))
        self.p_node = expected_density/nodes
        self.cyclic = None
        self.enforce_dag = enforce_dag

    def __call__(self):
        vertices = np.arange(self.nodes)
        for i in range(self.nodes):
            if self.enforce_dag:
                possible_parents = vertices[:i]
            else:
                possible_parents = np.setdiff1d(vertices, i)
            num_parents = np.random.binomial(n=len(possible_parents), p=self.p_node)
            parents = np.random.choice(possible_parents, size=num_parents, replace=False)

            # In networkx, the adjacency matrix is such that
            # the rows denote the parents and the columns denote the children. 
            # That is, W_ij = 1 ==> i -> j exists in the graph.
            self.adjacency_matrix[parents, i] = 1
            self.g = nx.DiGraph(self.adjacency_matrix)
            self.cyclic = not nx.is_directed_acyclic_graph(self.g)

        return self.g

class linearSEM:

    """
    -------------------------------------------------------------------
    This class models a Linear Structural Equation Model (Linear SEM)
    -------------------------------------------------------------------
    The model is initialized with the number of nodes in the graph and
    the absolute minimum and maximum weights for the edges. 
    """
    def __init__(self, graph, abs_weight_low=0.2, abs_weight_high=0.9, noise_scale=0.5, contractive=True):
        self.graph = graph
        self.abs_weight_low = abs_weight_low 
        self.abs_weight_high = abs_weight_high
        self.contractive = contractive

        self.n_nodes = len(graph.nodes)
        
        self.weights = np.random.uniform(self.abs_weight_low, self.abs_weight_high, size=(self.n_nodes, self.n_nodes))
        self.weights *= 2 * np.random.binomial(1, 0.5, size=self.weights.shape) - 1
        self.weights *= nx.to_numpy_array(self.graph)
        self.noise_scale = noise_scale

        if not self.contractive:
            self.weights = make_non_cotractive(self.weights)
        else:
            self.weights = make_contractive(self.weights)

    def generateData(self, n_samples, intervention_set=[None], lat_provided=False, latent_vec=None, fixed_intervention=False, return_latents=False):
        # set intervention_set = [None] for purely observational data.
        
        observed_set = np.setdiff1d(np.arange(self.n_nodes), intervention_set)
        U = np.zeros((self.n_nodes, self.n_nodes))
        U[observed_set, observed_set] = 1

        C = np.zeros((self.n_nodes, n_samples))
        if intervention_set[0] != None:
            if fixed_intervention:
                C[intervention_set, :] = np.random.randn(len(intervention_set), 1)
            else:
                C[intervention_set, :] = np.random.randn(len(intervention_set), n_samples)

        I = np.eye(self.n_nodes)
        if lat_provided:
            E = latent_vec.T
        else:
            E = self.noise_scale * np.random.randn(self.n_nodes, n_samples)
        X = np.linalg.inv(I - U @ self.weights.T) @ (U @ E + C)

        # The final data matrix is dimensions - n_samples X self.nodes
        if return_latents:
            return X.T, E.T
            
        return X.T

class Dataset:
    """
    -------------------------------------------------------------------------------------------------
    A class that stores the dataset parameters and generates the dataset.
    -------------------------------------------------------------------------------------------------
    Parameters:
    1)  n_nodes - (int) - number of nodes in the graph. 
    2)  expected_density - (int) - expected number of outgoing edges per node.
    3)  n_samples - (list of numbers) - number of samples in each experiment.
    4)  n_experiments - (int) - number of experiments to be performed. 
    5)  target_predef - (bool) - If True, then the target for each experiment has
                                 to be provided by the user. 
                                 Else, the targets are randomly selected in each 
                                 experiment. 
    6)  min_targets - (int) - minimum targets in each experiment.
    7)  max_targets - (int) - maximum targets in each experiment. 
    8)  mode - (string) - 'sat-pair-condition' - the targets are chosen such that
                                                 pair condition is satisfied.
                          'indiv-node' - Each experiment intervenes on a single node.
                                         Note that the pair condition is always satisfied 
                                         in this case. 
                          'no-constraint' - targets are chosen randomly in each experiment
                                            with no further constraints.
                          'block-node' - Each experiment intervenes on a fixed block of nodes
                                         and all nodes are covered. 
    9)  abs_weight_low - (frac) - absolute least value of the edge weights. 
    10) abs_weight_high - (frac) - absolute largest value of the edge weights.  
    11) targets - (list(list)) - list of targets for each experiments, None if target_predef=False.
    12) sem_type - (string) - 'linear' - sample from linear SEM.
                              'non-linear' - sample from non-linear SEM. 
    13) graph_provided - (bool) - True, if the graph is provided as an input.
    14) graph - (nx graph) - the graph definition the parent-child relations.
    15) gen_model_provided - (bool) - True, if the generative model is provided.
    16) gen_model - (gen model instance) - instance of the generative model (linearSEM/nonlinearSEM)

    Here, targets refer to the set of nodes intervened in an experiment. If the number of experiments
    is not sufficient to satisfy the requirement of the constraint, then the value is adjusted to 
    allow for the mode constraint to be satisfied. 
    """

    def __init__(
        self,
        n_nodes,
        expected_density,
        n_samples,
        n_experiments,
        target_predef=False, 
        min_targets=1,
        max_targets=4,
        mode='block-node',
        abs_weight_low=0.2,
        abs_weight_high=0.8,
        targets=None,
        graph_provided=False, 
        graph=None,
        gen_model_provided=False, 
        gen_model=None,
        noise_scale=0.5,
        enforce_dag=False,
        contractive=True
    ):
        self.n_nodes = n_nodes
        self.expected_density = expected_density
        self.n_samples = n_samples
        self.n_experiments = n_experiments
        self.target_predef = target_predef
        self.min_targets = min_targets
        self.max_targets = max_targets
        self.mode = mode
        self.abs_weight_low = abs_weight_low
        self.abs_weight_high = abs_weight_high
        self.noise_scale = noise_scale
        self.enforce_dag = enforce_dag
        self.contractive = contractive

        self.pair_condition_matrix = np.zeros((self.n_nodes, self.n_nodes)) # intervention (rows) X observations (column)

        if self.target_predef:
            # self.targets stores the set of nodes intervened in each experiment.
            self.targets = targets
            assert len(self.targets) == self.n_experiments, f"Expected len(targets) to be {n_experiments}, got {len(targets)}" 
            self.checkPairCondition()
        
        else:
            self._pick_targets()

        if graph_provided:
            self.graph = graph
        else:
            generator = DirectedGraphGenerator(nodes=self.n_nodes, expected_density=self.expected_density, enforce_dag=self.enforce_dag)
            self.graph = generator()
        
        if gen_model_provided:
            self.gen_model = gen_model
            if not graph_provided:
                self.graph = self.gen_model.graph
        else:
            self.gen_model = linearSEM(self.graph, self.abs_weight_low, self.abs_weight_high, noise_scale=self.noise_scale, contractive=contractive)

    def generate(self, interventions=True, change_targets=False, fixed_interventions=False, return_latents=False):
        dataset = list()
        if return_latents:
            latents = list()
        if interventions:

            if change_targets:
                self._pick_targets()

            for target_set in self.targets:
                data = self.gen_model.generateData(n_samples=self.n_samples, intervention_set=target_set, fixed_intervention=fixed_interventions, return_latents=return_latents)
                if return_latents:
                    dataset.append(data[0])
                    latents.append(data[1])
                else:
                    dataset.append(data)
        
        else:
            data = self.gen_model.generateData(n_samples=self.n_samples, intervention_set=[None], return_latents=return_latents)
            if return_latents:
                dataset.append(data[0])
                latents.append(data[1])
            else:
                dataset.append(data)

        if return_latents:
            return dataset, latents

        return dataset 
    
    def get_adjacency(self):
        return nx.to_numpy_array(self.graph)

    def _pick_targets(self, max_iterations=100000):
        
        iter = 0
        if self.mode not in ['indiv-node', 'sat-pair-condition', 'no-constraint', 'block-node']:
            print(f"{self.mode} does not exist, defaulting to 'indiv-node'")
            self.mode = 'indiv-node'

        if self.mode == 'indiv-node':
            assert self.n_experiments == self.n_nodes, f"expected {self.n_nodes}, got {self.n_experiments}"
            self.targets = [np.array([node]) for node in range(self.n_nodes)]
            self.pair_condition = True

        elif self.mode == 'block-node':
            assert self.n_experiments == self.n_nodes, f"expected {self.n_nodes}, got {self.n_experiments}"
            v_set = list(range(self.n_nodes))
            block_size = self.min_targets
            self.targets = [np.array([v_set[i-j] for i in range(block_size)]) for j in range(self.n_nodes)]
            self.pait_condition = True

        else:
            not_correct = True
            self.targets = list()
            for _ in range(self.n_experiments):
                iter += 1
                n_targets = np.random.randint(self.min_targets, self.max_targets+1, 1)
                target_set = np.random.choice(self.n_nodes, n_targets, replace=False)
                self.targets.append(target_set)

                observed_set = np.setdiff1d(np.arange(self.n_nodes), target_set)
                indices = np.ix_(target_set, observed_set)
                self.pair_condition_matrix[indices] = 1.0

            if not self.checkPairCondition() and self.mode == 'sat-pair-condition':
                for node in range(self.n_nodes):
                    if self.pair_condition_matrix[node, :].sum() != self.n_nodes-1:
                        self.targets.append(np.array([node]))
                
                self.n_experiments = len(self.targets)

            self.pair_condition = self.checkPairCondition()

    def checkPairCondition(self):
        return self.pair_condition_matrix.sum() == self.n_nodes**2 - self.n_nodes
