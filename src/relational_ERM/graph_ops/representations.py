import numpy as np


class PackedAdjacencyList:
    """ A structure representing a packed adjacency list. """

    def __init__(self, neighbours, weights, offsets, lengths, vertex_index):
        """ Initialize a new packed adjacency list

        Parameters
        ----------
        neighbours: an array representing the list of all neighbours.
        weights: an array of weights for each edge.
        offsets: the offset into the neighbours array for each vertex.
        lengths: the lengths of the subarray for the given vertex.
        vertex_index: an index mapping the current vertex index to the
            vertices in the full network.
        """
        self.neighbours = neighbours
        self.weights = weights
        self.offsets = offsets
        self.lengths = lengths
        self.vertex_index = vertex_index

    def get_neighbours(self, vertex):
        """ Get the list of neighbours of a given vertex.

        Parameters
        ----------
        vertex: the vertex for which to get the neighbours.
        """
        offset = self.offsets[vertex]
        length = self.lengths[vertex]
        return self.neighbours[offset:offset + length]

    def __len__(self):
        return len(self.lengths)


def redundant_edge_list_to_adj_list(edge_list, weights):
    """ Converts a redundant edge list to an adjacency list

    Parameters
    ----------
    edge_list: A numpy ndarray of dimension 2 representing the set of edges
    weights: A numpy ndarray of dimension 1 representing the weights for each edge

    Returns
    -------
    a dictionary of lists representing the adjacency list description of the graph.
    """
    el, w = np.copy(edge_list), np.copy(weights)

    # sort edge list
    asort = el[:, 0].argsort()
    el = el[asort]
    w = w[asort]

    verts = np.unique(el[:, 0])
    neighbour_dict = {}

    last_index = 0

    for user in verts:
        next_index = np.searchsorted(el[:, 0], user + 1)
        neighbours = el[last_index:next_index, 1]

        asort = neighbours.argsort()
        neighbour_dict[user] = (neighbours[asort], w[last_index:next_index][asort])

        last_index = next_index

    return neighbour_dict


def create_packed_adjacency_list(adjacency_list):
    """ Creates a packed adjacency list from a given adjacency list in the dictionary representation.

    Note that keys in the adjacency list are required to be contiguous from 0 to the number
    of vertices - 1.

    Parameters
    ----------
    adjacency_list: The adjacency list represented as a dictionary, where keys are vertices,
        and items are given by pairs of arrays representing the neighbours, and the corresponding
        weight associated with the connection to that neighbour.

    Returns
    -------
    packed_adjacency_list: A PackedAdjacencyList which represents the same graph.
    """
    num_vertex = len(adjacency_list)

    lengths = np.empty(num_vertex, dtype=np.int32)
    offsets = np.zeros(num_vertex, dtype=np.int32)
    neighbours_lists = []
    weights_lists = []

    for i in range(num_vertex):
        neighbours_i, weights_i = adjacency_list[i]
        neighbours_lists.append(neighbours_i)
        weights_lists.append(weights_i)
        lengths[i] = len(neighbours_i)

    neighbours = np.concatenate(neighbours_lists)
    weights = np.concatenate(weights_lists)
    np.cumsum(lengths[:-1], out=offsets[1:])

    return PackedAdjacencyList(neighbours, weights, offsets, lengths, np.arange(num_vertex))


def create_packed_adjacency_from_redundant_edge_list(redundant_edge_list):
    """ Creates a packed adjacency list from the given edge list.

    Parameters
    ----------
    redundant_edge_list: a two dimensional array containing the edge list.

    Returns
    -------
    packed_adjacency_list: the packed adjacency list corresponding to the given edge list.
    """
    idx = np.lexsort((redundant_edge_list[:, 1], redundant_edge_list[:, 0]))
    redundant_edge_list = redundant_edge_list[idx, :]
    vertices, counts = np.unique(redundant_edge_list[:, 0], return_counts=True)

    if vertices[0] != 0 or np.any(np.diff(vertices) != 1):
        raise ValueError("Source vertices do not form a contiguous range!")

    neighbours = np.require(redundant_edge_list[:, 1], dtype=np.int32, requirements='C')
    lengths = counts.astype(np.int32, copy=False)
    offsets = np.empty_like(lengths)
    np.cumsum(lengths[:-1], out=offsets[1:])
    offsets[0] = 0

    return PackedAdjacencyList(neighbours, None, offsets, lengths, np.arange(len(vertices), dtype=np.int32))


def adj_list_to_red_edge_list(adj_list):
    """ Converts an adjacency list to a redundant edge list.

    Params
    ------
    adjacency_list: The adjacency list represented as a dictionary
        of pairs of arrays, representing the neighbours and the weights.

    Returns
    -------
    edge_list: A two-dimensional arrays representing a redundant edge list.
    w: A one-dimensional array representing the weight associated with each edge.
    """

    el_one_list = []
    el_two_list = []
    w_list = []

    for vert, neighbours in adj_list.items():
        el_two, w = neighbours
        el_one = np.repeat(vert, el_two.shape[0])

        el_one_list += [el_one]
        el_two_list += [el_two]
        w_list += [w]

    el_one = np.concatenate(el_one_list)
    el_two = np.concatenate(el_two_list)
    el = np.stack([el_one, el_two], 1)

    w = np.concatenate(w_list)

    return el, w


def packed_adj_list_to_red_edge_list(packed_adj_list: PackedAdjacencyList):
    """ Converts a packed adjacency list to a redundant edge list.

    Params
    ------
    packed_adj_list: the adjacency list to convert

    Returns
    -------
    edge_list: A two-dimensional arrays representing a redundant edge list.
    weights: A one-dimensional array representing the weight associated with each edge.
    """
    edge_list = np.empty((len(packed_adj_list.neighbours), 2), dtype=packed_adj_list.neighbours.dtype)
    weights = np.copy(packed_adj_list.weights)

    edge_list[:, 0] = np.repeat(np.arange(len(packed_adj_list.lengths), dtype=packed_adj_list.lengths.dtype),
                                packed_adj_list.lengths)
    edge_list[:, 1] = packed_adj_list.neighbours

    return edge_list, weights


def adj_list_to_edge_list(adj_list):
    """
    Takes an adjacency list corresponding to an undirected graph and returns the edge list

    :param adj_list:
    :return:
    """
    red_el, w = adj_list_to_red_edge_list(adj_list)

    c_maj = red_el[:, 0] <= red_el[:, 1]

    return red_el[c_maj], w[c_maj]


def edge_list_to_adj_list(edge_list, weights):
    el, w = edge_list_to_red_edge_list(edge_list, weights)
    return redundant_edge_list_to_adj_list(el, w)


def edge_list_to_red_edge_list(edge_list, weights):
    el_flip = np.stack([edge_list[:, 1], edge_list[:, 0]], axis=1)
    no_diag = (el_flip[:,0] != el_flip[:,1])

    return np.concatenate([edge_list, el_flip[no_diag]]), \
           np.concatenate([weights, weights[no_diag]])


def red_edge_list_to_edge_list(red_edge_list, weights):
    c_maj = (red_edge_list[:, 0] <= red_edge_list[:, 1])
    return red_edge_list[c_maj], weights[c_maj]


def directed_to_undirected(edge_list, weights):
    rel, rw = edge_list_to_red_edge_list(edge_list, weights)
    return red_edge_list_to_edge_list(rel, rw)


def relabel(edge_list):
    shape = edge_list.shape
    vertex_index, edge_list = np.unique(edge_list, return_inverse=True)
    edge_list = edge_list.astype(np.int32).reshape(shape)

    return edge_list, vertex_index
