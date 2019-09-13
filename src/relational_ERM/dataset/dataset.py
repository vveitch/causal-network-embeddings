"""

TODO: replace 'label' w treatment / outcome / confounder

"""
import argparse
import time

import numpy as np
import tensorflow as tf

try:
    import mkl_random as random
except ImportError:
    import numpy.random as random

from relational_ERM.sampling import adapters, factories
from relational_ERM.data_cleaning.pokec import load_data_pokec, process_pokec_attributes


def add_parser_sampling_arguments(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--proportion-censored', type=float, default=0.5,
                        help='proportion of censored vertex labels at train time.')
    parser.add_argument('--batch-size', type=int, default=1, help='minibatch size')
    parser.add_argument('--dataset-shards', type=int, default=None, help='dataset parallelism')

    parser.add_argument('--sampler', type=str, default=None, choices=factories.dataset_names(),
                        help='the sampler to use')

    parser.add_argument('--sampler-test', type=str, default=None, choices=factories.dataset_names(),
                        help='if not None, the sampler to use for testing')

    parser.add_argument('--num-edges', type=int, default=800,
                        help='Number of edges per sample.')

    parser.add_argument('--window-size', type=int, default=10,
                        help='Context size for optimization. Default is 10.')

    parser.add_argument('--num-negative', type=int, default=5,
                        help='negative examples per vertex for negative sampling')

    parser.add_argument('--num-negative-total', type=int, default=None,
                        help='total number of negative vertices sampled')

    return parser


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


def get_dataset_fn(sampler, args):
    if sampler is None:
        sampler = 'biased-walk'

    return factories.make_dataset(sampler, args)


def make_input_fn(graph_data, args, treatments, outcomes, dataset_fn=None, num_samples=None):
    def input_fn():

        dataset = dataset_fn(graph_data, args.seed)

        data_processing = adapters.compose(
            adapters.relabel_subgraph(),  # 0-index the sampled subgraph
            adapters.append_vertex_labels(treatments, 'treatment'),
            adapters.append_vertex_labels(outcomes, 'outcome'),
            adapters.make_split_vertex_labels(
                graph_data.num_vertices, args.proportion_censored,
                np.random.RandomState(args.seed)),
            adapters.add_sample_size_info(),
            adapters.format_features_labels())

        dataset = dataset.map(data_processing, 8)
        if num_samples is not None:
            dataset = dataset.take(num_samples)

        batch_size = args.batch_size

        if batch_size is not None:
            dataset = dataset.apply(
                adapters.padded_batch_samples(batch_size, t_dtype=treatments.dtype, o_dtype=outcomes.dtype))

        dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

        return dataset

    return input_fn


def make_no_graph_input_fn(graph_data, args, treatments, outcomes, filter_test=False):
    """
    A dataset w/ all the label processing, but no graph structure.
    Used at evaluation and prediction time

    """
    def input_fn():

        vertex_dataset = tf.data.Dataset.from_tensor_slices(
            ({'vertex_index': np.expand_dims(np.array(range(graph_data.num_vertices)), 1),
                     'is_positive': np.expand_dims(np.array(range(graph_data.num_vertices)), 1)},))

        data_processing = adapters.compose(
            adapters.append_vertex_labels(treatments, 'treatment'),
            adapters.append_vertex_labels(outcomes, 'outcome'),
            adapters.make_split_vertex_labels(
                graph_data.num_vertices, args.proportion_censored,
                np.random.RandomState(args.seed)),
            adapters.format_features_labels())

        dataset = vertex_dataset.map(data_processing, 8)

        if filter_test:
            def filter_test_fn(features, labels):
                return tf.equal(tf.squeeze(features['in_test']), 1)

            dataset = dataset.filter(filter_test_fn)

        batch_size = args.batch_size
        dataset = dataset.batch(batch_size=batch_size, drop_remainder=False)

        return dataset

    return input_fn


def main():
    session_config = tf.ConfigProto(intra_op_parallelism_threads=0, inter_op_parallelism_threads=4)
    # session_config = tf.ConfigProto()
    tf.enable_eager_execution(config=session_config)

    print("This is the RERM (network) dataset test")

    parser = add_parser_sampling_arguments()
    args = parser.parse_args()

    print("load the data")
    graph_data, profiles = load_data_pokec('../dat/pokec/regional_subset')

    print("number of edges {}".format(graph_data.edge_list.shape[0]))

    pokec_features = process_pokec_attributes(profiles)

    treatments = pokec_features['I_like_books']
    outcomes = pokec_features['relation_to_casual_sex']

    print("make the graph sampler")
    dataset_fn_train = get_dataset_fn(args.sampler, args)

    # dataset = dataset_fn_train(graph_data, args.seed)
    # itr = dataset.make_one_shot_iterator()
    # t0 = time.time()
    # for _ in range(1000):
    #     sample = itr.get_next()
    # t1 = time.time()
    # print(t1-t0)

    print("make the input_fn")
    # make_sample_generator = make_input_fn(graph_data, args, treatments, outcomes, dataset_fn_train)
    make_sample_generator = make_no_graph_input_fn(graph_data, args, treatments, outcomes, filter_test=True)
    sample_generator = make_sample_generator()

    itr = sample_generator.make_one_shot_iterator()

    in_treat_and_test = []
    in_treat = []

    t0 = time.time()
    for _ in range(1000):
        sample = itr.get_next()
        treatment = sample[0]['treatment']
        test = sample[0]['in_test']
        in_treat += [np.mean(treatment)]
        in_treat_and_test += [np.sum(tf.cast(treatment, tf.float32)*test)]
    t1 = time.time()
    print(t1-t0)

    print(np.mean(in_treat))
    print(np.mean(in_treat_and_test))

    print(tf.equal(tf.squeeze(sample[0]['in_test']), 1))

    print(sample[0].keys())
    print(sample[0].keys())
    print(sample[0]['treatment'].shape[0].value)
    # print(sample[1])
    # print(sample[0]['vertex_index'])
    # print(np.max(sample[0]['vertex_index']))


if __name__ == "__main__":
    main()