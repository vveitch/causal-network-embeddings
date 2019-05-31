import numbers

import numpy as np
import tensorflow as tf


def make_learned_unigram_logits(num_vertex, power=0.75, prior=None, accumulator_dtype=tf.int32, name=None):
    """ Compute unigram distribution from the empirical vertex distribution.

    Note: this function creates graph elements when called. Only call it along the graph
    corresponding to the dataset this will be used for.

    Parameters
    ----------
    num_vertex: the number of vertices in the graph.
    power: a power to raise the empirical distribution to.
    prior: prior distribution of counts. Either an integer, to set constant prior counts,
        or a vector of length `num_vertex`, representing prior counts for each vertex.
    accumulator_dtype: the type to use for accumulating the empirical distribution.
    name: an optional name for the operation.

    Returns
    -------
    make_logits: constructs the logits from the given empirical distribution,
        and updates the empirical distribution.
    """
    if prior is None:
        prior = 100

    if isinstance(prior, numbers.Number):
        prior = np.ones(num_vertex, dtype=accumulator_dtype) * prior

    empirical_vertex_distribution = tf.get_variable(
        name='empirical_vertex_distribution',
        dtype=accumulator_dtype,
        initializer=prior,
        trainable=False,
        use_resource=True)

    def make_logits(edge_list):
        with tf.name_scope(name, "learned_unigram_distribution", [edge_list]):
            edge_list_flat = tf.reshape(edge_list, [-1])
            update_empirical = tf.scatter_add(
                empirical_vertex_distribution,
                edge_list_flat,
                tf.ones_like(edge_list_flat))

            with tf.control_dependencies([update_empirical]):
                logits = power * tf.log(tf.to_float(empirical_vertex_distribution))
        return logits

    return make_logits


def add_negative_sample(num_vertices,
                        num_samples_per_vertex,
                        num_random_total=None,
                        vertex_distribution_logit=None,
                        bias_by_vertex_occurrence=False,
                        seed=None):
    """ Adapts an existing sampler to add uniform negative samples.

    Parameters
    ----------
    num_vertices: The number of vertices in the graph.
    num_samples_per_vertex: The number of random vertices to sample for each one in the graph.
    num_random_total: The number of random vertices to sample in total.
    vertex_distribution_logit: If not None, negative examples are selected according
        to this distribution, which represents the unnormalized log-probabilities
    bias_by_vertex_occurrence: Whether to bias the number of negative examples produced for a given
        vertex by the number of occurrences of that vertex in the edge list. It seems
        that the canonical setting is bias_by_degree=False.
    seed: the seed to use.

    Returns
    -------
    fn: an adapter that can be mapped across a dataset to add uniform negative examples.
    """

    def _sample_independent_vertices(edge_list, num_samples, dtype):
        if vertex_distribution_logit is None:
            return tf.random_uniform(
                tf.reshape(num_samples, [1]),
                0, num_vertices, seed=seed, dtype=dtype)

        if callable(vertex_distribution_logit):
            unigram_logits = vertex_distribution_logit(edge_list)
        else:
            unigram_logits = vertex_distribution_logit

        return tf.multinomial(tf.expand_dims(unigram_logits, 0),
                              num_samples,
                              seed=seed,
                              output_dtype=dtype)

    def _make_neg_edges_shared(edge_list, neg_edges_shape, dtype):
        """ Creates negative edges where negative vertices are drawn from a fixed pool for
        the sample. This is somewhat more computationally efficient.
        """
        random_vertex_ids = _sample_independent_vertices(edge_list, num_random_total, dtype)
        neg_edges_end_idx = tf.random_uniform(
            neg_edges_shape, 0, num_random_total,
            seed=seed,
            dtype=tf.int32)

        neg_edges_end = tf.gather(random_vertex_ids, neg_edges_end_idx)
        return neg_edges_end

    def _make_neg_edges_indep(edge_list, neg_edges_shape, dtype):
        """ Creates negative edges where negative vertices are drawn at random from
        all vertices in the graph. This is less computationally efficient.
        """
        neg_edges_end = _sample_independent_vertices(edge_list, tf.reduce_prod(neg_edges_shape), dtype)
        neg_edges_end = tf.reshape(neg_edges_end, neg_edges_shape)

        return neg_edges_end

    if num_random_total is not None:
        _make_neg_edges_end = _make_neg_edges_shared
    else:
        _make_neg_edges_end = _make_neg_edges_indep

    def fn_tensorflow(edge_list, weights):
        vertex_list, _ = tf.unique(tf.reshape(edge_list, [-1]))

        if bias_by_vertex_occurrence:
            neg_edges_start = tf.tile(edge_list[:, 0], [num_samples_per_vertex])
        else:
            neg_edges_start = tf.tile(vertex_list, [num_samples_per_vertex])

        if weights is None:
            weights = tf.ones(tf.stack([tf.shape(edge_list)[0], 1]))

        neg_edges_end = _make_neg_edges_end(edge_list, tf.shape(neg_edges_start), edge_list.dtype)
        neg_edges = tf.stack([neg_edges_start, neg_edges_end], axis=1)
        neg_weights = tf.zeros(tf.stack([tf.shape(neg_edges)[0], tf.shape(weights)[1]]))

        all_edges = tf.concat([edge_list, neg_edges], axis=0)
        all_weights = tf.concat([weights, neg_weights], axis=0)

        return all_edges, all_weights, vertex_list

    def fn(data):
        edge_list = data['edge_list']
        weights = data.get('weights', None)

        edge_list, weights, positive_vertices = fn_tensorflow(edge_list, weights)

        return {**data, 'edge_list': edge_list, 'weights': weights, 'positive_vertices': positive_vertices}

    return fn
