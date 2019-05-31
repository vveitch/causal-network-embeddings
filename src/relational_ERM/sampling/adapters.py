""" This module provides functions to adapt sub-samples from one sampler to the common format.

Our current input format for estimators is given by the following tuple:

For features:
edge_list: a 2-dimensional tensor as a redundant edge list, where each row represents
    an edge by its vertex index.
weights: 1-dimensional tensor with a weight associated for each edge.
vertex_index: a 1-dimensional tensor, mapping the vertex indices in the current
    subsample to indices in the entire graph.

For labels
labels: the labels associated with each labelled vertex.

"""

import numpy as np
import tensorflow as tf

import relational_ERM.tensorflow_ops.array_ops
from relational_ERM.tensorflow_ops import adapter_ops as tensorflow_adapters, array_ops
from relational_ERM.graph_ops import representations

try:
    import mkl_random as random
except ImportError:
    import numpy.random as random


def apply_adapter(sampler, adapter):
    """ Applies an adapter to the given sampler

    Parameters
    ----------
    sampler: a sampling function to adapt.
    adapter: the adapter to apply to the sampling function.

    Returns
    -------
    sample: a sampling function
    """

    def sample():
        data = sampler()
        return adapter(data)

    return sample


def compose(*fns):
    """ Composes the given functions in reverse order.

    Parameters
    ----------
    fns: the functions to compose

    Returns
    -------
    comp: a function that represents the composition of the given functions.
    """
    import functools

    def _apply(x, f):
        if isinstance(x, tuple):
            return f(*x)
        else:
            return f(x)

    def comp(*args):
        return functools.reduce(_apply, fns, args)

    return comp


def get_edge_index(graph: representations.PackedAdjacencyList, edge_list, vertex_index=None):
    """ Gets the index of each edge in the given edge list.

    Parameters
    ----------
    graph: the packed adjacency list representing the entire graph.
    edge_list: a two dimensional array representing edges as pairs of vertices
    vertex_index: an optional array mapping the indices from the edges to indices
        in the full graph.
    """
    if vertex_index is None:
        vertex_index = range(len(graph.offsets))

    edge_index = np.empty(edge_list.shape[0], dtype=edge_list.dtype)

    for i, (s, t) in enumerate(edge_list):
        s = vertex_index[s]
        t = vertex_index[t]
        edge_index[i] = graph.offsets[s] + np.searchsorted(graph.get_neighbours(s), t)

    return edge_index


def adapt_random_walk():
    """ This function adapts a random walk sampler which produces a vertex list into
    a function returning the adequate structure to be fed into estimators.

    Returns
    -------
    fn: a function that takes
    """

    def fn_numpy(walk):
        num_edges = len(walk) - 1

        # fill in the redundant edge list
        edge_list = np.empty((num_edges, 2), dtype=np.int32)

        edge_list[:num_edges, 0] = walk[:-1]
        edge_list[:num_edges, 1] = walk[1:]
        return edge_list

    def fn_tensorflow(walk):
        edge_list_start = walk[:-1]
        edge_list_end = walk[1:]

        return tf.stack([edge_list_start, edge_list_end], axis=1)

    def fn(data):
        walk = data['walk']

        if isinstance(walk, tf.Tensor):
            edge_list = fn_tensorflow(walk)
        else:
            edge_list = fn_numpy(walk)

        return {**data, 'edge_list': edge_list}

    return fn


def adapt_random_walk_window(window_size):
    """ Adapt a random walk with a window size.

    This adapter turns a random walk into an edge list, where
    all the edges in a window of the given walk are included.

    Parameters
    ----------
    window_size: An integer representing the size of the window.

    Returns
    -------
    fn: the windowing transformation.
    """
    if window_size <= 0:
        raise ValueError('The window size must be at least 1.')

    def fn(data):
        walk = data['walk']

        edge_lists = []

        for i in range(1, window_size + 1):
            edge_lists.append(
                tf.stack([walk[:-i], walk[i:]], axis=1)
            )

        edge_list = tf.concat(edge_lists, axis=0)

        return {**data, 'edge_list': edge_list}

    return fn


def adapt_random_walk_induced(neighbours, lengths, offsets, name=None):
    """ Adapts a random walk to produce the induced subgraph.

    Parameters
    ----------
    neighbours: the packed adjacency list of the full graph.
    lengths: the lengths of each subarray in neighbours.
    offsets: the offsets into the neighbours array.
    name: the name of the operation to create.

    Returns
    -------
    fn: a transformation that reads the data from the `walk` key
        and produced a `neighbours`, `lengths` and `offsets` key
        defining the packed subgraph.
    """

    def fn(data):
        walk = data['walk']

        with tf.name_scope(name, 'random_walk_to_induced', [walk, neighbours, lengths, offsets]):
            vertex_index = tf.contrib.framework.sort(
                tf.unique(walk)[0],
                name='walk_index')

            subgraph = tensorflow_adapters.get_induced_subgraph(
                vertex_index,
                neighbours=neighbours,
                lengths=lengths,
                offsets=offsets)

        return {
            **data,
            'neighbours': subgraph[0],
            'lengths': subgraph[1],
            'offsets': subgraph[2],
            'vertex_index': vertex_index
        }

    return fn


def adapt_packed_subgraph():
    """ Gets an edge list representation from a packed sample representation.

    This function adapts a stream of elements that represent a packed adjacency
    list, contained in the keys:
    - vertex_index: the index mapping from local vertex index to global vertex index
    - neighbours: packed adjacency list in neighbours array
    - lengths: lengths of sub-arrays in packed adjacency list

    Returns
    -------
    fn: a function which transforms a packed sample to an edge list.
    """

    def fn(data):
        vertex_index = data['vertex_index']

        edge_list = tensorflow_adapters.adjacency_to_edge_list(data['neighbours'], data['lengths'])
        edge_list = tf.gather(vertex_index, edge_list)

        del data['vertex_index']

        return {
            **data,
            'edge_list': edge_list,
        }

    return fn


def adapt_packed_subgraph_posneg(num_neg_per_pos=None, seed=None):
    """ Gets an edge list representation from a packed sample representation
    containing both edges and (a subsample) of non-edges.

    Parameters
    ----------
    num_neg_per_pos: if not None, the number of negative edges to keep
        for every positive edge. If None, keep all the negative edges.
    seed: the seed to use for random subsampling.

    Returns
    -------
    fn: a function which transforms a packed sample to an edge list, and
        a list of non-edges.
    """

    def fn(data):
        vertex_index = data['vertex_index']

        edge_list_pos, edge_list_neg = tensorflow_adapters.adjacency_to_posneg_edge_list(
            data['neighbours'], data['lengths'], redundant=False)
        edge_list_pos = tf.gather(vertex_index, edge_list_pos, name='RestoreIndexPos')
        edge_list_neg = tf.gather(vertex_index, edge_list_neg, name='RestoreIndexNeg')

        if num_neg_per_pos is not None:
            num_pos = tf.shape(edge_list_pos)[0]
            num_neg = tf.to_int32(num_neg_per_pos * num_pos)

            neg_idx = tf.random_uniform(
                tf.reshape(num_neg, [1]),
                minval=0, maxval=tf.shape(edge_list_neg)[0],
                dtype=tf.int32, seed=seed)

            edge_list_neg = tf.gather(edge_list_neg, neg_idx, name='SubsampleNegEdges')

        del data['vertex_index']

        weights_pos = tf.ones(tf.reshape(tf.shape(edge_list_pos)[0], [1]), dtype=tf.float32)
        weights_neg = tf.zeros(tf.reshape(tf.shape(edge_list_neg)[0], [1]), dtype=tf.float32)

        edge_list = tf.concat([edge_list_pos, edge_list_neg], axis=0)
        weights = tf.concat([weights_pos, weights_neg], axis=0)

        return {
            **data,
            'edge_list': edge_list,
            'weights': weights
        }

    return fn


def relabel_subgraph():
    """ This function adapts an existing sampler by relabelling the vertices in the edge list
    to have dense index.

    Returns
    -------
    sample: a function, that when invoked, produces a sample for the input function.
    """

    def relabel(edge_list, positive_vertices):
        shape = edge_list.shape
        vertex_index, edge_list = np.unique(edge_list, return_inverse=True)
        edge_list = edge_list.astype(np.int32).reshape(shape)

        # relabel the positive vertices
        positive_verts = np.searchsorted(vertex_index, positive_vertices)
        is_positive = np.zeros_like(vertex_index)
        is_positive[positive_verts] = 1

        return edge_list, vertex_index, is_positive

    def sample(data):
        edge_list = data['edge_list']
        positive_vertices = data.get('positive_vertices', tf.unique(tf.reshape(edge_list, [-1]))[0])
        vertex_index = data.get('vertex_index', None)

        if isinstance(edge_list, tf.Tensor):
            new_edge_list, new_vertex_index, is_positive = tf.py_func(relabel, [edge_list, positive_vertices],
                                                                      [tf.int32, tf.int32, tf.int32], stateful=False)
            new_edge_list.set_shape(edge_list.shape)
            new_vertex_index.set_shape([None])
            is_positive.set_shape([None])
        else:
            new_edge_list, new_vertex_index, is_positive = relabel(edge_list, positive_vertices)

        if vertex_index is not None:
            if isinstance(vertex_index, tf.Tensor):
                vertex_index = tf.gather(vertex_index, new_vertex_index, name='resample_vertex_index')
            else:
                vertex_index = vertex_index[new_vertex_index]
        else:
            vertex_index = new_vertex_index

        return {**data, 'edge_list': new_edge_list, 'vertex_index': vertex_index, 'is_positive': is_positive}

    return sample


def append_vertex_labels(labels, label_name='labels'):
    """ Adapts an existing sampler to append labels.

    This function adapts the given sampler by appending slices of labels
    to the sample corresponding to the given vertex.

    Parameters
    ----------
    labels: the labels for each vertex.
    """

    def fn(data):
        vertex_index = data['vertex_index']

        if isinstance(vertex_index, tf.Tensor):
            sample_labels = tf.gather(labels, vertex_index, axis=0)
        else:
            sample_labels = labels[vertex_index, :]

        return {**data, label_name: sample_labels}

    return fn


def append_packed_vertex_labels(packed_labels, lengths, offsets=None):
    """ Adapts an existing sampler to append label information.

    In many cases, we may wish to consider a variable number of labels per
    vertex, in which case one-hot encodings may be very wasteful. In this function,
    we instead consider simply fetching a variable length list.

    Parameters
    ----------
    packed_labels: a 1-dimensional array, representing all labels for all vertices concatenated.
    lengths: a 1-dimensional array, representing the length of the subarray.
    offsets: a 1-dimensional array, representing the offsets of the subarray. If None, computed
        from the lengths subarray.
    """
    if offsets is None:
        offsets = np.empty_like(lengths)
        np.cumsum(lengths[:-1], out=offsets[1:])
        offsets[0] = 0

    def fn(data):
        vertex_index = data['vertex_index']

        subset_lengths = tf.gather(lengths, vertex_index)

        vertex_labels = relational_ERM.tensorflow_ops.array_ops.concatenate_slices(
            packed_labels,
            tf.gather(offsets, vertex_index),
            subset_lengths)

        return {
            **data,
            'packed_labels': vertex_labels,
            'packed_labels_lengths': subset_lengths,
        }

    return fn


def append_sparse_vertex_classes(classes):
    """ Adapts an existing sampler to append classes

    Parameters
    ----------
    classes: the labels for each vertex.
    """

    # hack to stop tensorflow from storing the full array in the graph def
    def _make_hidden_constant(value, name):
        return tf.py_func(
            lambda: value,
            [], tf.int32, stateful=False,
            name=name)

    all_node_classes = _make_hidden_constant(classes, "make_all_node_classes")
    all_node_classes.set_shape(classes.shape)

    def fn(data):
        vertex_index = data['vertex_index']

        if isinstance(vertex_index, tf.Tensor):
            sample_classes = tf.gather(all_node_classes, vertex_index)
        else:
            sample_classes = all_node_classes[vertex_index]

        return {**data, 'classes': sample_classes}

    return fn


def append_vertex_vector_features(vector_features):
    """ Adapts an existing sampler to append vector-valued features

    This function adapts the given sampler by appending slices of
    a features array to the sample containing the given vertex.

    Parameters
    ----------
    vector_features: the features for all vertices.
    """
    # hack to stop tensorflow from storing the full array in the graph def
    def _make_features_hidden_constant(value, name):
        return tf.py_func(
            lambda: value,
            [], tf.float32, stateful=False,
            name=name)

    all_node_features = _make_features_hidden_constant(vector_features, "create_all_node_features")
    all_node_features.set_shape(vector_features.shape)

    def fn(data):
        vertex_index = data['vertex_index']

        if isinstance(vertex_index, tf.Tensor):
            sample_features = tf.gather(all_node_features, vertex_index, axis=0)
        else:
            sample_features = all_node_features[vertex_index, :]

        return {**data, 'vertex_features': sample_features}

    return fn


def format_features_labels():
    """ The tensorflow estimator structures take in two arguments intended to represent
    the features and the labels of a given problem. We use two dictionaries with attributes
    attached to the features and labels. This function splits our internal representation
    to that representation

    Returns
    -------
    fn: a function that can be applied using the map function.
    """

    feature_keys = [
        'edge_list', 'weights', 'vertex_index', 'num_edges', 'num_vertex', 'is_positive', 'vertex_features']

    # we require label information during predict, so gotta change this as a hacky workaround
    feature_keys = feature_keys + [
        'labels', 'treatment', 'outcome',
        'split', 'in_test', 'in_dev', 'in_train']

    # label_keys = [
    #     'labels', 'treatment', 'outcome',
    #     'split', 'in_test', 'in_dev', 'in_train',
    #     'packed_labels', 'packed_labels_lengths', 'packed_labels_indices',
    #     'classes']

    def fn(data):
        features = {k: v for k, v in data.items() if k in feature_keys}
        # labels = {k: v for k, v in data.items() if k in label_keys}
        labels = {}

        return features, labels

    return fn


def add_sample_size_info():
    """ Add batch size information to the samples. This is useful for recovering
    information after batching.

    Returns
    -------
    fn: a function to map onto a dataset to append shape information.
    """
    def fn(data):
        shape_info = {
            'num_edges': tf.shape(data['edge_list'])[0],
            'num_vertex': tf.size(data['vertex_index'])
        }

        return {**data, **shape_info}

    return fn


def padded_batch_samples(batch_size, t_dtype=np.int32, o_dtype=np.int32):
    feature_pad_values = {
        'edge_list': 0,
        'weights': -1.0,
        'vertex_index': 0,
        'num_edges': 0,
        'num_vertex': 0,
        'is_positive': -1
    }

    def fn(dataset):
        if 'vertex_features' in dataset.output_shapes[0]:
            feature_pad_values['vertex_features'] = 0.0

        label_pad_values = {
        }

        if 'in_test' in dataset.output_shapes[0]:
            label_pad_values['in_test'] = 0.

        if 'in_train' in dataset.output_shapes[0]:
            label_pad_values['in_train'] = 0.

        if 'in_dev' in dataset.output_shapes[0]:
            label_pad_values['in_dev'] = 0.

        if 'treatment' in dataset.output_shapes[0]:
            # dense labels
            label_pad_values['treatment'] = np.zeros([], dtype=t_dtype)
        if 'outcome' in dataset.output_shapes[0]:
            # dense labels
            label_pad_values['outcome'] = np.zeros([], dtype=o_dtype)

        feature_pad_values.update(label_pad_values)

        return dataset.padded_batch(
            batch_size, dataset.output_shapes,
            (feature_pad_values, {}),
            drop_remainder=True)

    return fn


def split_vertex_labels(num_vertices, proportion_censored, rng=None):
    """ Adapts tensorflow dataset to produce another element in the labels dictionary
    corresponding to whether the vertex is in the training or testing set.

    Parameters
    ----------
    num_vertices: The number of vertices in the graph.
    proportion_censored: The proportion of graph to censor
    rng: An instance of np.random.RandomState used to split the data. If None,
        a deterministic split will be chosen (corresponding to seeding with 42).

    Returns
    -------
    fn: A function that can be used to map a dataset to censor some of the vertex labels.
    """
    if rng is None:
        rng = random.RandomState(42)

    split = rng.binomial(1, 1 - proportion_censored, size=num_vertices).astype(np.float32)

    def fn(data):
        vertex_id = data['vertex_index']
        sample_split = tf.gather(split, vertex_id)

        return {**data, 'split': sample_split}

    return fn


def make_split_vertex_labels(num_vertices, proportion_censored, rng=None):
    """ Adapts tensorflow dataset to produce another element in the labels dictionary
    corresponding to whether the vertex is in the training or testing set.

    Parameters
    ----------
    num_vertices: The number of vertices in the graph.
    proportion_censored: The proportion of graph to censor
    rng: An instance of np.random.RandomState used to split the data. If None,
        a deterministic split will be chosen (corresponding to seeding with 42).

    Returns
    -------
    fn: A function that can be used to map a dataset to censor some of the vertex labels.
    """
    if rng is None:
        rng = random.RandomState(42)

    # todo: can have overlap in dev and test sets... fix me
    in_test = rng.binomial(1, proportion_censored, size=num_vertices).astype(np.float32)
    in_dev = rng.binomial(1, proportion_censored, size=num_vertices).astype(np.float32)
    in_train = ((in_test+in_dev)==0).astype(np.float32)

    def fn(data):
        vertex_id = data['vertex_index']
        sample_in_test = tf.gather(in_test, vertex_id)
        sample_in_dev = tf.gather(in_dev, vertex_id)
        sample_in_train = tf.gather(in_train, vertex_id)

        return {**data, 'in_test': sample_in_test, 'in_dev': sample_in_dev, 'in_train': sample_in_train}

    return fn