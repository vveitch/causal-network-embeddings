""" Factory to create canned datasets.

This module implements a variety of canned datasets which implement a reasonable
strategy to produce a set of positive and negative examples (edges).

"""

import tensorflow as tf
import numpy as np

import relational_ERM.sampling.negative_sampling as negative_sampling
from . import adapters
from ..tensorflow_ops import adapter_ops, dataset_ops


def parallel_dataset(dataset_fn, num_shards, seed):
    """ Builds the dataset to pull in parallel by using parallel interleaved datasets.

    Parameters
    ----------
    dataset_fn: a function which creates a dataset with the given seed.
    num_shards: the number of shards for each dataset.
    seed: the seed to use.

    Returns
    -------
    dataset: a dataset pulling in parallel from the given number of shards.
    """
    # from tensorflow.data.experimental import parallel_interleave
    from tensorflow.contrib.data import parallel_interleave

    if num_shards is None or num_shards == 1:
        return dataset_fn(seed)

    print('parallel dataset seems ok...')
    print(num_shards)
    seed_offset_dataset = tf.data.Dataset.range(num_shards)
    seed_offset_dataset = seed_offset_dataset.repeat()
    seed_offset_dataset = seed_offset_dataset.shuffle(num_shards)
    return seed_offset_dataset.apply(
        parallel_interleave(
            lambda input: dataset_fn(seed + input),
            cycle_length=num_shards))
            # ,
            # block_length=1, sloppy=True, buffer_output_elements=None))


def get_unigram_distribution(num_vertex, dataset_fn, power=0.75):
    """ Compute the empirical unigram distribution from the given dataset.

    This function returns unnormalized logits for the unigram distribution
    raised to the given power.

    Parameters
    ----------
    num_vertex: the number of vertices in the graph.
    dataset_fn: a nullary function creating the dataset.
    power: a power to raise the unigram distribution to.

    Returns
    -------
    logits: the logits for the given distribution.
    """
    tf.logging.info("Computing unigram distribution from sample.")

    with tf.Graph().as_default():
        dataset = dataset_fn()
        samples = dataset.take(10*num_vertex)
        sample = samples.make_one_shot_iterator().get_next()

        if 'walk' in sample:
            sample = sample['walk']
        else:
            sample = sample['vertex_index']

        empirical_unigram = tf.get_variable(
            'empirical_unigram', shape=[num_vertex], dtype=tf.int32,
            initializer=tf.zeros_initializer(), trainable=False)

        increment_empirical = tf.scatter_add(
            empirical_unigram, sample, tf.ones_like(sample, dtype=tf.int32))

        initializer = tf.global_variables_initializer()

        with tf.Session() as session:
            session.run(initializer)
            for _ in range(10*num_vertex):
                session.run(increment_empirical)

            vertex_count_empirical = session.run(empirical_unigram)

    tf.logging.info("Done computing unigram distribution.")
    return power*np.log(vertex_count_empirical)


def get_negative_sample(graph_data, args, seed=None, unigram_distribution=None):
    """ Obtain the unigram negative sampler corresponding to the given graph.

    Parameters
    ----------
    graph_data: A graph_data structure.
    args: hyperparameters for the negative sampling.
    seed: if not None, the seed to use to seed the negative sampling.
    unigram_distribution: if given, unnormalized logits for the distribution to sample from,
        otherwise uses a learnt unigram distribution.

    Returns
    -------
    add_negative_sample: a function which produces negative samples.
    """
    if unigram_distribution is None:
        unigram_distribution = negative_sampling.make_learned_unigram_logits(
            graph_data.num_vertices, prior=graph_data.adjacency_list.lengths)

    add_negative_sample = negative_sampling.add_negative_sample(
        graph_data.num_vertices,
        num_samples_per_vertex=args.num_negative,
        num_random_total=args.num_negative_total,
        vertex_distribution_logit=unigram_distribution,
        seed=seed)
    return add_negative_sample


def tensorboard_hack(graph_data):
    """
    When graph_data is used to construct a dataset the contents get stored as a constant tensor.
    This is stupid, and causes tensorboard to shit itself.
    This function is a hack to prevent this behaviour.

    Returns: neighbours, lengths, offsets

    remark: could alternatively define a variable for each array and assign to this variable
    """

    def _constant_hidden_value(value, name):
        return tf.py_func(
            lambda: value,
            [], tf.int32, stateful=False,
            name=name)

    adjacency_list = graph_data.adjacency_list

    neighbours = _constant_hidden_value(adjacency_list.neighbours, 'create_neighbours')
    lengths = _constant_hidden_value(adjacency_list.lengths, 'create_lengths')
    offsets = _constant_hidden_value(adjacency_list.offsets, 'create_offsets')

    return neighbours, lengths, offsets


def make_biased_random_walk_dataset(args):
    """ DeepWalk-style random walk dataset, with unigram negative samples. """
    def dataset_fn(graph_data, seed):
        # adjacency_list = graph_data.adjacency_list
        # neighbours = tf.constant(adjacency_list.neighbours, dtype=tf.int32)
        # lengths = tf.constant(adjacency_list.lengths, dtype=tf.int32)
        # offsets = tf.constant(adjacency_list.offsets, dtype=tf.int32)

        neighbours, lengths, offsets = tensorboard_hack(graph_data)

        def _fn(s):
            return dataset_ops.RandomWalkDataset(
                int(args.num_edges / args.window_size),
                neighbours, lengths, offsets, seed=s)

        add_negative_sample = get_negative_sample(
            graph_data, args, seed)

        dataset = parallel_dataset(_fn, args.dataset_shards, seed)

        dataset = dataset.map(
            adapters.compose(
                adapters.adapt_random_walk_window(args.window_size),
                add_negative_sample),
            num_parallel_calls=args.dataset_shards)

        return dataset

    return dataset_fn


def make_psample_negative_dataset(args):
    """ P-sampled dataset, with unigram negative samples. """
    def dataset_fn(graph_data, seed):
        neighbours, lengths, offsets = tensorboard_hack(graph_data)

        def _fn(s):
            return dataset_ops.PSamplingDataset(
                args.num_edges, neighbours, lengths, offsets, seed=s)

        add_negative_sample = get_negative_sample(graph_data, args, seed)

        dataset = parallel_dataset(_fn, args.dataset_shards, seed)

        dataset = dataset.map(
            adapters.compose(
                adapters.adapt_packed_subgraph(),
                add_negative_sample),
            num_parallel_calls=args.dataset_shards)

        return dataset

    return dataset_fn


def make_uniform_edge_dataset(args):
    """ Uniform edge-sampled dataset. """
    def dataset_fn(graph_data, seed):
        neighbours, lengths, offsets = tensorboard_hack(graph_data)

        def _fn(s):
            return dataset_ops.UniformEdgeDataset(
                args.num_edges, neighbours=neighbours, lengths=lengths, offsets=offsets, seed=s)

        dataset = parallel_dataset(_fn, args.dataset_shards, seed)
        return dataset.map(get_negative_sample(graph_data, args, seed),
                           num_parallel_calls=args.dataset_shards)

    return dataset_fn


def make_psample_induced_dataset(args):
    """ P-sampled dataset, with subsampled negative edges from induced subgraph. """
    def dataset_fn(graph_data, seed):
        neighbours, lengths, offsets = tensorboard_hack(graph_data)

        def _fn(s):
            return dataset_ops.PSamplingDataset(args.num_edges, neighbours, lengths, offsets, seed=s)

        dataset = parallel_dataset(_fn, args.dataset_shards, seed)

        dataset = dataset.map(
            adapters.adapt_packed_subgraph_posneg(args.num_negative),
            num_parallel_calls=args.dataset_shards)

        return dataset

    return dataset_fn


def make_biased_walk_induced_dataset(args):
    """ Deepwalk-style dataset, with subsampled negative edges from the induced subgraph. """
    def dataset_fn(graph_data, seed):
        neighbours, lengths, offsets = tensorboard_hack(graph_data)

        def _fn(s):
            return dataset_ops.RandomWalkDataset(
                args.num_edges // args.window_size,
                neighbours=neighbours, lengths=lengths, offsets=offsets, seed=s)

        dataset = parallel_dataset(_fn, args.dataset_shards, seed)

        dataset = dataset.map(
            adapters.compose(
                adapters.adapt_random_walk_induced(neighbours, lengths, offsets),
                adapters.adapt_packed_subgraph_posneg(args.num_negative)
            ),
            num_parallel_calls=args.dataset_shards)

        return dataset

    return dataset_fn


def make_biased_walk_induced_pos_dataset(args):
    """ Deepwalk-style dataset, with positive edges from induced subgraph and uniform negative edges """
    def dataset_fn(graph_data, seed):
        neighbours, lengths, offsets = tensorboard_hack(graph_data)

        def _fn(s):
            return dataset_ops.RandomWalkDataset(
                args.num_edges // args.window_size,
                neighbours=neighbours, lengths=lengths, offsets=offsets, seed=s)

        dataset = parallel_dataset(_fn, args.dataset_shards, seed)
        add_negative_sample = get_negative_sample(graph_data, args, seed)

        dataset = dataset.map(
            adapters.compose(
                adapters.adapt_random_walk_induced(neighbours, lengths, offsets),
                adapters.adapt_packed_subgraph(),
                add_negative_sample
            ),
            num_parallel_calls=args.dataset_shards)

        return dataset

    return dataset_fn


def make_open_ego_dataset(args):
    """ Open ego sampling, with centers uniformly sampled, and unigram negative samples. """
    def dataset_fn(graph_data, seed):
        num_vertex_per_sample = int(args.num_edges / np.mean(graph_data.adjacency_list.lengths))
        num_vertex_graph = len(graph_data.adjacency_list.lengths)

        dataset = tf.data.Dataset.range(1).repeat()

        dataset = dataset.map(
            adapters.compose(
                lambda _: tf.random_uniform([num_vertex_per_sample], 0, num_vertex_graph, dtype=tf.int32, seed=seed),
                lambda centers: {'edge_list': adapter_ops.get_open_ego_network(
                    centers,
                    graph_data.adjacency_list.neighbours,
                    graph_data.adjacency_list.lengths,
                    graph_data.adjacency_list.offsets)},
                get_negative_sample(graph_data, args, seed)),
            num_parallel_calls=args.dataset_shards)

        return dataset

    return dataset_fn


_dataset_factories = {
    'biased-walk': make_biased_random_walk_dataset,
    'p-sampling': make_psample_negative_dataset,
    'p-sampling-induced': make_psample_induced_dataset,
    'biased-walk-induced': make_biased_walk_induced_dataset,
    'biased-walk-induced-uniform': make_biased_walk_induced_pos_dataset,
    'ego-open': make_open_ego_dataset,
    'uniform-edge': make_uniform_edge_dataset
}


def make_dataset(name, args):
    """ Creates the dataset with the given name.

    Parameters
    ----------
    name: the name of the dataset to create.
    args: hyper-paremeters for the dataset.

    Returns
    -------
    dataset_fn: a function which creates the specified dataset.
    """
    if name not in _dataset_factories:
        raise ValueError("Unknown sampler type.")

    return _dataset_factories[name](args)


def dataset_names():
    """ Gets valid names for the datasets.

    Returns
    -------
    a list of strings representing valid dataset names.
    """
    return list(_dataset_factories.keys())
