import tensorflow as tf


def edge_list_to_adj_mat(n_vert, edge_list, weights=None, force_simple=True):
    """
    Convert edge list (of undirected graph) to adjacency matrix

    :param n_vert: int, number of vertices
    :param edge_list: [2, num_edges], 0-indexed
    :param weights: [num_edges, 1] (default: all 1.)
    :param force_simple: boolean, forces the return to correspond to a simple graph with loops
    :return: [n_vert, n_vert] tensor w dtype = weights.dtype
    """

    swapped_el = tf.stack([edge_list[:,1], edge_list[:,0]], axis=1)
    indices = tf.concat([edge_list, swapped_el], axis=0)

    if weights is not None:
        updates = tf.squeeze(tf.concat([weights, weights], axis=0))
    else:
        updates = tf.ones_like(indices[:, 0], dtype=tf.float32)

    adj_mat = tf.scatter_nd(
        indices,
        updates,
        [n_vert, n_vert],
    )

    if force_simple:
        simple_indices = tf.cast(tf.where(tf.not_equal(adj_mat, 0)), dtype=tf.int32)
        simple_updates = tf.ones(tf.shape(simple_indices)[0], dtype=tf.float32)
        adj_mat = tf.scatter_nd(
            simple_indices,
            simple_updates,
            [n_vert, n_vert],
        )

    return adj_mat