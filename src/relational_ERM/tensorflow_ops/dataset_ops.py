import tensorflow as tf
import distutils.sysconfig as du_config
import os
import numbers

from tensorflow.python.framework import random_seed


_loaded_library = None


def _library():
    global _loaded_library

    if _loaded_library is None:
        directory = os.path.dirname(__file__)
        library_filename = '_datasets_tensorflow' + du_config.get_config_var('EXT_SUFFIX')
        path = os.path.join(directory, library_filename)
        _loaded_library = tf.load_op_library(path)

    return _loaded_library


class BiasedRandomWalkDataset(tf.data.Dataset):
    """ This class implements a node2vec-style biased random walk sampler. """
    def __init__(self, walk_length, p, q, neighbours, lengths, offsets, seed=0):
        """ Initializes a new instance of the biased walk sampler.

        Parameters
        ----------
        walk_length: a scalar tensor of type int32 indicating the length of the walk to sample.
        p: The return parameter.
        q: The in-out parameter.
        neighbours: a rank-1 tensor of type int32 indicating the neighbours of each vertex,
        lengths: a rank-1 tensor of type int32 indicating the length of the subarray in
            neighbours corresponding to a given vertex.
        offsets: a rank-1 tensor of type int32 indicating the offsets of the subarry in
            neighbours corresponding to a given vertex.
        seed: an integer value representing the seed.
        """
        self.seed, self.seed2 = random_seed.get_seed(seed)
        self.neighbours = neighbours
        self.lengths = lengths
        self.offsets = offsets
        self.walk_length = walk_length
        self.p = p
        self.q = q
        super(BiasedRandomWalkDataset, self).__init__()

    def _as_variant_tensor(self):
        return _library().biased_walk_dataset(
            seed=self.seed,
            seed2=self.seed2,
            walk_length=self.walk_length,
            p=self.p, q=self.q,
            neighbours=self.neighbours,
            lengths=self.lengths,
            offsets=self.offsets)

    @property
    def output_types(self):
        return {'walk': tf.int32}

    @property
    def output_shapes(self):
        return {'walk': tf.TensorShape([None])}

    @property
    def output_classes(self):
        return {'walk': tf.Tensor}


class PSamplingDataset(tf.data.Dataset):
    """ This class implements a p-sampler on the given dataset. """
    def __init__(self, n, neighbours, lengths, offsets, seed=0):
        """ Initializes a new instance of the p-sampling dataset.

        Parameters
        ----------
        n: The average number of edges in a given sample.
        neighbours: a rank-1 tensor of type int32 indicating the neighbours of each vertex,
        lengths: a rank-1 tensor of type int32 indicating the length of the subarray in
            neighbours corresponding to a given vertex.
        offsets: a rank-1 tensor of type int32 indicating the offsets of the subarry in
            neighbours corresponding to a given vertex.
        seed: an integer value representing the seed.
        """
        self.seed, self.seed2 = random_seed.get_seed(seed)
        self.neighbours = neighbours
        self.offsets = offsets
        self.lengths = lengths

        if tf.contrib.framework.is_tensor(neighbours):
            self.p = tf.sqrt(n / tf.to_float(tf.size(neighbours)))
        else:
            self.p = tf.sqrt(n / len(neighbours))

        super(PSamplingDataset, self).__init__()

    def _as_variant_tensor(self):
        return _library().p_sampling_dataset(
            seed=self.seed,
            seed2=self.seed2,
            p=self.p,
            neighbours=self.neighbours,
            lengths=self.lengths,
            offsets=self.offsets)

    @property
    def output_types(self):
        return {
            'lengths': tf.int32,
            'neighbours': tf.int32,
            'offsets': tf.int32,
            'vertex_index': tf.int32
        }

    @property
    def output_shapes(self):
        return {
            'neighbours': tf.TensorShape([None]),
            'lengths': tf.TensorShape([None]),
            'offsets': tf.TensorShape([None]),
            'vertex_index': tf.TensorShape([None])
        }

    @property
    def output_classes(self):
        return {
            'lengths': tf.Tensor,
            'neighbours': tf.Tensor,
            'offsets': tf.Tensor,
            'vertex_index': tf.Tensor
        }


class UniformEdgeDataset(tf.data.Dataset):
    """ This class implements a uniform edge sampler. """
    def __init__(self, sample_size, neighbours, lengths, offsets, seed=0):
        """ Initializes a new instance of the uniform edge sampler.

        Parameters
        ----------
        sample_size: The number of edges in each sample.
        neighbours: The array of neighbours.
        lengths: The array of lengths in the neighbours.
        offsets: The array of offsets into the neighbours.
        seed: The random seed to use.
        """
        self.sample_size = sample_size
        self.seed, self.seed2 = random_seed.get_seed(seed)
        self.neighbours = neighbours
        self.lengths = lengths
        self.offsets = offsets
        super(UniformEdgeDataset, self).__init__()

    def _as_variant_tensor(self):
        return _library().uniform_edge_dataset(
            seed=self.seed,
            seed2=self.seed2,
            n=self.sample_size,
            neighbours=self.neighbours,
            lengths=self.lengths,
            offsets=self.offsets)

    @property
    def output_types(self):
        return {'edge_list': tf.int32}

    @property
    def output_shapes(self):
        if isinstance(self.sample_size, numbers.Number):
            num_samples = self.sample_size
        else:
            num_samples = None

        return {'edge_list': tf.TensorShape([num_samples, 2])}

    @property
    def output_classes(self):
        return {'edge_list': tf.Tensor}


class RandomWalkDataset(tf.data.Dataset):
    """ This class implements a uniform random walk on a graph. """
    def __init__(self, walk_length, neighbours, lengths, offsets, seed=0):
        """ Initialize a new random walk dataset.

        Parameters
        ----------
        walk_length: a scalar tensor representing the length of the walk.
        neighbours: a 1-dimensional tensor representing the packed adjacency list.
        lengths: a 1-dimensional tensor representing the subarrays in neighbours.
        offsets: a 1-dimensional tensor representing the subarrays in neighbours.
        seed: the seed to use.
        """
        self.walk_length = walk_length
        self.seed, self.seed2 = random_seed.get_seed(seed)
        self.neighbours = neighbours
        self.lengths = lengths
        self.offsets = offsets
        super(RandomWalkDataset, self).__init__()

    def _as_variant_tensor(self):
        return _library().random_walk_dataset(
            seed=self.seed,
            seed2=self.seed2,
            walk_length=self.walk_length,
            neighbours=self.neighbours,
            lengths=self.lengths,
            offsets=self.offsets)

    @property
    def output_types(self):
        return {'walk': tf.int32}

    @property
    def output_shapes(self):
        if isinstance(self.walk_length, numbers.Number):
            walk_length = self.walk_length
        else:
            walk_length = None

        return {'walk': tf.TensorShape([walk_length])}

    @property
    def output_classes(self):
        return {'walk': tf.Tensor}

