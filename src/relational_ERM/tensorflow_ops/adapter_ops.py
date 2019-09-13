import os
import distutils.sysconfig as du_config
import tensorflow as tf

_loaded_library = None


def _library():
    global _loaded_library

    if _loaded_library is None:
        directory = os.path.dirname(__file__)
        library_filename = '_adapters_tensorflow' + du_config.get_config_var('EXT_SUFFIX')
        path = os.path.join(directory, library_filename)
        _loaded_library = tf.load_op_library(path)

    return _loaded_library


def adjacency_to_edge_list(neighbours, lengths, redundant=True):
    """ Converts a packed adjacency list into an edge list of positive edges.

    Parameter
    ---------
    neighbours: a tensor representing the list of neighbours in packed format.
    lengths: a tensor representing the lengths of each subarray in the list of neighbours.
    redundant: if true, no assumption is made on the adjacency list and the obtained
        edge list is redundant, that is, both edges (a, b) and (b, a) appear in the list.
        Otherwise, the adjacency list is assumed to be symmetric, and the obtained edge list
        is canonical, containing only edges (a, b) with a < b.

    Returns
    -------
    edge_list: a tensor of shape [len(neighbours), 2], representing the set of positive
        edges in the given sample.
    """
    return _library().adjacency_to_edge_list(
        neighbours=neighbours, lengths=lengths, redundant=redundant)


def adjacency_to_posneg_edge_list(neighbours, lengths, redundant=True):
    """ Converts a packed adjacency list into a pair of edge list of positive and
    negative edges.

    Parameter
    ---------
    neighbours: a tensor representing the list of neighbours in packed format.
    lengths: a tensor representing the lengths of each subarray in the list of neighbours.
    redundant: if true, no assumption is made on the adjacency list and the obtained
        edge list is redundant, that is, both edges (a, b) and (b, a) appear in the list.
        Otherwise, the adjacency list is assumed to be symmetric, and the obtained edge list
        is canonical, containing only edges (a, b) with a < b.

    Returns
    -------
    edge_list_pos: a tensor of shape [len(neighbours), 2] representing the set of positive
        edges in the given sample.
    edge_list_neg: a tensor representing the set of negative edges in the given sample.
    """
    return _library().adjacency_to_pos_neg_edge_list(
        neighbours=neighbours, lengths=lengths, redundant=redundant)


def get_induced_subgraph(vertex, neighbours, lengths, offsets):
    """ Gets the induced subgraph on the given vertex.

    This function computes the induced subgraph for a given set of
    vertices. The vertices given in `vertex` must be unique.

    Parameters
    ----------
    vertex: a 1-dimensional tensor representing the list of vertices
        for which to obtain an induced subgraph.
    neighbours: a 1-dimensional tensor representing the packed adjacency list
        of the full graph.
    lengths: a 1-dimensional tensor representing the lengths of each subarray
        in neighbours.
    offsets: a 1-dimensional tensor representing the offsets of each subarray
        in neighbours.

    Returns
    -------
    neighbours: a 1-dimensional tensor representing the packed adjacency list for the induced
        subgraph.
    lengths: a 1-dimensional tensor representing the lengths of each subarray in neighbours.
    offsets: a 1-dimensional tensor representing the offsets of each subarray in neighbours.
    """
    return _library().get_induced_subgraph(
        vertex=vertex, neighbours=neighbours, lengths=lengths, offsets=offsets)


def get_open_ego_network(centers, neighbours, lengths, offsets):
    """ Gets the induced ego sample produced by the given centers.

    This function computes an edge list returning all edges that are
    attached to the given set of centers.

    Parameters
    ----------
    centers: the vertices from which to obtain the graphs.
    neighbours: a 1-dimensional tensor representing the packed adjacency list.
    lengths: a 1-dimensional tensor representing the lengths of each subarray.
    offsets: a 1-dimensional tensor representing the offsets of each subarray.

    Returns
    -------
    edge_list: a 2-dimensional tensor representing an edge list.
    """
    return _library().get_open_ego_network(
        centers=centers, neighbours=neighbours, lengths=lengths, offsets=offsets)
