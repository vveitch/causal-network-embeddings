import os
import distutils.sysconfig as du_config
import tensorflow as tf

_loaded_library = None


def _library():
    global _loaded_library

    if _loaded_library is None:
        directory = os.path.dirname(__file__)
        library_filename = '_array_ops_tensorflow' + du_config.get_config_var('EXT_SUFFIX')
        path = os.path.join(directory, library_filename)
        _loaded_library = tf.load_op_library(path)

    return _loaded_library


def concatenate_slices(input_, begins, sizes):
    """ Extracts and concatenates slices from the given variable.

    Parameters
    ----------
    input_: a 1-dimensional tensor from which to extract slices.
    begins: a 1-dimensional tensor, representing the start of the slices to extract.
    sizes: a 1-dimensional tensor, representing the lengths of the slices to extract.

    Returns
    -------
    concat_slices: a 1-dimensional tensor representing concatenated slices.
    """
    return _library().concatenate_slices(input_, begins, sizes)


def packed_to_sparse_index(lengths):
    """ Converts a packed array to an index array describing
    the nonzero locations.

    Parameters
    ----------
    lengths: a 1-dimensional tensor representing the lengths of each subarray.

    Returns
    -------
    indices: a 2-dimensional tensor representing the indices.
    """
    return _library().packed_to_sparse_index(lengths)


def repeat(values, counts):
    """ Repeats elements of the values array according to counts.

    Parameters
    ----------
    values: a 1-dimensional tensor representing values to repeat.
    counts: a scalar or vector tensor representing the numeber of times to repeat
        each value.

    Returns
    -------
    a 1-dimensional tensor representing the repeated values.
    """
    return _library().repeat(values, counts)


def batch_length_to_segment(lengths, output_columns):
    return _library().batch_length_to_segment(lengths, output_columns)
