from relational_ERM.graph_ops.representations import edge_list_to_adj_list, redundant_edge_list_to_adj_list
import networkx as nx
import numpy as np
import scipy as sp


def is_connected(edge_list):
    V = np.unique(edge_list).shape[0]
    as_csr = sp.sparse.csr_matrix((np.ones_like(edge_list[:, 0]), (edge_list[:, 0], edge_list[:, 1])), [V,V])
    G = nx.from_scipy_sparse_matrix(as_csr)
    return nx.is_connected(G)