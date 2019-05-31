import collections
import tensorflow as tf
import re


def _get_value(value_or_fn):
    if callable(value_or_fn):
        return value_or_fn()
    else:
        return value_or_fn


def _default_embedding_optimizer():
    # embedding optimization


    # word2vec decays linearly to a min learning rate (default: 0.0001), decreasing each "epoch"
    # however, node2vec and deepwalk run only 1 "epoch" each

    # learning_rate = tf.train.polynomial_decay(
    #     10.,
    #     global_step,
    #     100000,
    #     end_learning_rate=0.0001,
    #     power=1.0,
    #     cycle=False,
    #     name="Word2Vec_decay"
    # )

    # gensim word2vec default learning rate is 0.025
    return tf.train.GradientDescentOptimizer(learning_rate=0.025)


def _default_global_optimizer():
    # return tf.train.RMSPropOptimizer(learning_rate=5e-4, momentum=0.9)
    global_step = tf.train.get_or_create_global_step()
    # learning_rate = tf.train.polynomial_decay(
    #     10.,
    #     global_step,
    #     1000000,
    #     end_learning_rate=0.01,
    #     power=1.0,
    #     cycle=False,
    #     name="global_linear_decay"
    # )
    learning_rate = 1.
    return tf.train.GradientDescentOptimizer(learning_rate)


def _make_embedding_variable(params):
    embedding_variable_name = 'input_layer/vertex_index_embedding/embedding_weights'

    all_embeddings = tf.get_variable(
        embedding_variable_name,
        shape=[params['num_vertices'], params['embedding_dim']],
        dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=1 / params['embedding_dim']),
        trainable=params.get('embedding_trainable', True))
    # if params.get('embedding_checkpoint', None) is not None:
    #     tf.train.init_from_checkpoint(
    #         params['embedding_checkpoint'],
    #         {embedding_variable_name: all_embeddings})
    return all_embeddings


def make_simple_skipgram_loss(clip=None):
    """ Makes a simple skipgram loss for edge prediction from a given edge list.

    This function takes a simple edge list and does not further modify it. In particular,
    it does not apply any transformation such as windowing or pruning.

    Parameters
    ----------
    clip: If not None, a value to clip the individual losses at.

    Returns
    -------
    loss: a function which computes the loss.
    """
    def loss(edge_logits, num_vertex, edge_list, edge_weights, params):
        with tf.name_scope('skipgram_loss', values=[edge_logits, edge_list, edge_weights]):
            if len(edge_list.shape) == 3:
                batch_size = tf.to_float(tf.shape(edge_list)[0])
            else:
                batch_size = 1.

            edge_present = tf.to_float(tf.equal(edge_weights, 1))

            # values of -1 in the weights indicate padded edges which should be ignored
            # in loss computation.
            edge_censored = tf.to_float(tf.not_equal(edge_weights, -1))

            edge_pred_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=edge_present, logits=edge_logits)

            edge_pred_loss = edge_pred_loss * edge_censored

            if clip:
                edge_pred_loss = tf.clip_by_value(edge_pred_loss, 0, clip)

            # sum instead of (tf default of) mean because mean screws up learning rates for embeddings
            loss_value = tf.divide(tf.reduce_sum(edge_pred_loss), batch_size,
                                   name='skipgram_edge_loss')
        return loss_value

    return loss


def get_assignment_map_from_checkpoint(tvars, init_checkpoint, name_to_variable=None):
    """Compute the union of the current variables and checkpoint variables."""
    assignment_map = {}
    initialized_variable_names = {}

    name_to_variable2 = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable2[name] = var

    if name_to_variable is not None:
        print("DOESNT WORK")
        print(name_to_variable)
        print("DOES WORK")
        print(name_to_variable2)
    else:
        name_to_variable = name_to_variable2

    init_vars = tf.train.list_variables(init_checkpoint)

    assignment_map = collections.OrderedDict()
    for x in init_vars:
        (name, var) = (x[0], x[1])
        if name not in name_to_variable:
            print(name)
            continue
        assignment_map[name] = name
        initialized_variable_names[name] = 1
        initialized_variable_names[name + ":0"] = 1

    return (assignment_map, initialized_variable_names)


        # def _make_polyak_averaging(embeddings, features, label_logits, mode, polyak, make_label_logits, params):
#     batch_size = params['batch_size']
#     decay = 0.99
#
#     if batch_size is not None:
#         #  Adjust decay for batch size to take into account the minibatching.
#         decay = decay ** batch_size
#
#     label_ema = tf.train.ExponentialMovingAverage(decay=decay)
#     if polyak:
#         # predict logits by replacing the rerm_model params by a moving average
#         def label_ema_getter(getter, name, *args, **kwargs):
#             var = getter(name, *args, **kwargs)
#             ema_var = label_ema.average(var)
#             return ema_var  # if ema_var else var
#
#         # create the running average variable
#         label_ema_op = label_ema.apply(tf.global_variables("label_logits"))
#         with tf.control_dependencies([label_ema_op]):
#             with tf.variable_scope("label_logits", reuse=True, custom_getter=label_ema_getter):
#                 label_logits_predict = make_label_logits(embeddings, features, mode, params)
#     else:
#         # no polyak averaging; default behaviour
#         label_logits_predict = label_logits
#         label_ema_op = tf.no_op(name='no_polyak_averaging')
#
#     return label_ema_op, label_logits_predict