import numpy
import torch
import networkx as nx
from scipy.sparse import coo_matrix

def addedge(graph, num_edges_to_add):
    """
    Add edges to the graph to enhance the data.
    
    Args:
    - graph (nx.Graph): The original graph.
    - num_edges_to_add (int): Number of edges to add.

    Returns:
    - torch.Tensor: The enhanced adjacency matrix as a torch tensor.
    """
    nodes = list(graph.nodes())
    num_nodes = len(nodes)

    # Add edges
    for _ in range(num_edges_to_add):
        # Randomly select two nodes
        u, v = np.random.choice(num_nodes, 2, replace=False)
        # Add an edge between these nodes
        graph.add_edge(nodes[u], nodes[v])

    # Create adjacency matrix
    adj_matrix = nx.adjacency_matrix(graph)
    
    # Convert to torch tensor
    adj_tensor = torch.FloatTensor(adj_matrix.toarray())

    return adj_tensor