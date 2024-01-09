import numpy as np
from scipy.sparse import csr_matrix, identity
from scipy.sparse.linalg import inv,expm
from scipy.linalg import fractional_matrix_power


def compute_ppr(csr_graph: csr_matrix, alpha=0.2, self_loop=True):
    if self_loop:
        csr_graph = csr_graph + identity(csr_graph.shape[0])
    d = csr_graph.sum(axis=1).A1
    d_inv_sqrt = np.reciprocal(np.sqrt(d))
    d_inv_sqrt_mat = csr_matrix(np.diag(d_inv_sqrt))
    at = d_inv_sqrt_mat @ csr_graph @ d_inv_sqrt_mat
    return alpha * inv(identity(csr_graph.shape[0]) - (1 - alpha) * at)


def compute_heat(csr_graph: csr_matrix, t=5, self_loop=True):
    if self_loop:
        csr_graph = csr_graph + identity(csr_graph.shape[0])
    d = csr_graph.sum(axis=1).A1
    d_inv = np.reciprocal(d)
    d_inv_mat = csr_matrix(np.diag(d_inv))
    ad = csr_graph.dot(d_inv_mat)
    heat_kernel = expm(t * (ad - identity(csr_graph.shape[0])))
    return heat_kernel

# def decrease_edge()