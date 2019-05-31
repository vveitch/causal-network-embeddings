import sys
import tensorflow as tf

from relational_ERM.rerm_model.helpers import _default_global_optimizer, _default_embedding_optimizer, _make_embedding_variable, \
    make_simple_skipgram_loss, _get_value, get_assignment_map_from_checkpoint
from relational_ERM.rerm_model.logging import make_label_prediction_summaries, label_eval_metric_fn, unsupervised_eval_metric_fn


def _get_getter(ema):
    def ema_getter(getter, name, *args, **kwargs):
        var = getter(name, *args, **kwargs)
        ema_var = ema.average(var)
        return ema_var  # if ema_var else var
    return ema_getter


def _targeted_regularization(q0, q1, g, t, y, split):
    t_float = tf.cast(t, tf.float32)
    q = (1.-t_float)*q0 + t_float*q1

    h = t_float * (1. / tf.expand_dims(g,1)) + (1.-t_float) * (1. / (1.-tf.expand_dims(g,1)))

    epsilon = tf.get_variable('epsilon', [1])
    q_mod = q + epsilon * h

    loss = tf.losses.mean_squared_error(y, q_mod, weights=split, reduction=tf.losses.Reduction.SUM)
    return loss


def _make_feedforward_regressor(embedding, labels, split, num_hidden_layers):
    regularizer = tf.contrib.layers.l2_regularizer(scale=1e-6)

    if num_hidden_layers == 0:
        output = tf.layers.dense(embedding, 1, activation=None,
                                 kernel_regularizer=regularizer, bias_regularizer=regularizer)
    else:
        layer = tf.layers.dense(embedding, 200, activation=tf.nn.elu)
        for _ in range(num_hidden_layers - 1):
            layer = tf.layers.dense(layer, 200, activation=tf.nn.elu,
                                    kernel_regularizer=regularizer, bias_regularizer=regularizer)
        output = tf.layers.dense(layer, 1, activation=None,
                                 kernel_regularizer=regularizer, bias_regularizer=regularizer)

    with tf.name_scope("loss"):
        output = tf.squeeze(output)
        per_example_loss = tf.losses.mean_squared_error(labels, output, reduction=tf.losses.Reduction.NONE)
        loss = tf.losses.mean_squared_error(labels, output, weights=split, reduction=tf.losses.Reduction.SUM)

    # append nonsense to agree in shape w/ classifier output
    return loss, per_example_loss, None, output


def _make_feedforward_classifier(embedding, labels, num_labels, split, num_hidden_layers, label_smoothing=0.001):
    regularizer = tf.contrib.layers.l2_regularizer(scale=1e-6)

    if num_hidden_layers == 0:
        logits = tf.layers.dense(embedding, num_labels, activation=None,
                                 kernel_regularizer=regularizer, bias_regularizer=regularizer)
    else:
        layer = tf.layers.dense(embedding, 200, activation=tf.nn.elu)
        for _ in range(num_hidden_layers - 1):
            layer = tf.layers.dense(layer, 200, activation=tf.nn.elu,
                                    kernel_regularizer=regularizer, bias_regularizer=regularizer)
        logits = tf.layers.dense(layer, num_labels, activation=None,
                                 kernel_regularizer=regularizer, bias_regularizer=regularizer)

    with tf.name_scope("loss"):
        one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float32,
                                    on_value=1. - label_smoothing, off_value=label_smoothing)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        censored_per_example_loss = split * per_example_loss
        loss = tf.reduce_sum(censored_per_example_loss)

    probabilities = tf.nn.softmax(logits, axis=-1)[:, 0, 1]  # P(T=1)

    return loss, per_example_loss, logits, probabilities


def _create_or_get_dragonnet_rerm(embedding, is_training, treatment, outcome, split,
                                  treatment_cat=True, outcome_cat=True, getter=None):
    """
    Make predictions for the outcome, using the treatment and embedding,
    and predictions for the treatment, using the embedding
    Both outcome and treatment are assumed to be binary

    The main difference between this and the function used for PeerRead is that here we return the loss
    as the *mean* over the batch size instead of as the sum.

    This is stupider for training dynamics (requires a learning rate adjustment),
    but is required for the unsupervised and supervised loss to operate on the same scale

    Parameters
    ----------
    bert
    is_training
    treatment
    outcome
    label_dict
    split
    treatment_cat indicator for whether treatment is categorical
    outcome_cat indicator for whether outcome is categorical
    getter custom getter, for polyak averaging support

    Returns
    -------

    """

    treatment_float = tf.cast(treatment, tf.float32)
    with tf.variable_scope('dragon_net', reuse=tf.AUTO_REUSE, custom_getter=getter):
        with tf.variable_scope('treatment'):
            if treatment_cat:
                loss_t, per_example_loss_t, logits_t, expectation_t = _make_feedforward_classifier(
                    embedding, treatment, 2, split, num_hidden_layers=0)
            else:
                # logits is just None
                loss_t, per_example_loss_t, logits_t, expectation_t = _make_feedforward_regressor(
                    embedding, treatment, split, num_hidden_layers=0)

        with tf.variable_scope('outcome_st_treatment'):
            if outcome_cat:
                loss_ot1, per_example_loss_ot1, logits_ot1, expectation_ot1 = _make_feedforward_classifier(
                    embedding, outcome, 2, split=split*treatment_float, num_hidden_layers=2)
            else:
                loss_ot1, per_example_loss_ot1, logits_ot1, expectation_ot1 = _make_feedforward_regressor(
                    embedding, outcome, split=split*treatment_float, num_hidden_layers=2)

        with tf.variable_scope('outcome_st_no_treatment'):
            if outcome_cat:
                loss_ot0, per_example_loss_ot0, logits_ot0, expectation_ot0 = _make_feedforward_classifier(
                    embedding, outcome, 2, split=split*(1.-treatment_float), num_hidden_layers=2)
            else:
                loss_ot0, per_example_loss_ot0, logits_ot0, expectation_ot0 = _make_feedforward_regressor(
                    embedding, outcome, split=split*(1.-treatment_float), num_hidden_layers=2)

        # tar_reg = _targeted_regularization(expectation_ot0, expectation_ot1, expectation_t, treatment, outcome, split)

    training_loss = loss_ot0 + loss_ot1 + loss_t

    if is_training:
        batch_size = treatment.shape[0].value
        tf.losses.add_loss(loss_ot0 / batch_size)
        tf.losses.add_loss(loss_ot1 / batch_size)
        tf.losses.add_loss(loss_t / batch_size)

        training_loss = training_loss / batch_size

    outcome_st_treat = {'per_example_loss': per_example_loss_ot1,
                        'logits': logits_ot1,
                        'expectations': expectation_ot1}

    outcome_st_no_treat = {'per_example_loss': per_example_loss_ot0,
                           'logits': logits_ot0,
                           'expectations': expectation_ot0}

    treat = {'per_example_loss': per_example_loss_t,
             'logits': logits_t,
             'expectations': expectation_t}

    return training_loss, outcome_st_treat, outcome_st_no_treat, treat


def _make_treatment_outcome_eval_metric_fn(treatment_cat, outcome_cat):

    def _treatment_outcome_eval_metric_fn(outcome, treatment, in_test,
                                          per_example_loss_ot1, logits_ot1,
                                          per_example_loss_ot0, logits_ot0,
                                          per_example_loss_t, logits_t):
        treatment_float = tf.cast(treatment, tf.float32)

        ot1_dict = label_eval_metric_fn(per_example_loss_ot1, outcome, logits_ot1,
                                        split=treatment_float * in_test, family='outcome_given_treatment',
                                        is_categorical=outcome_cat)
        ot0_dict = label_eval_metric_fn(per_example_loss_ot0, outcome, logits_ot0,
                                        split=(1 - treatment_float) * in_test, family='outcome_given_no_treatment',
                                        is_categorical=outcome_cat)
        t_dict = label_eval_metric_fn(per_example_loss_t, treatment, logits_t,
                                      split=in_test, family='treatment',
                                      is_categorical=treatment_cat)

        return {**ot1_dict, **ot0_dict, **t_dict}

    return _treatment_outcome_eval_metric_fn


def _make_edge_logits(embeddings, features, edge_list, weights, params):
    """ Helper function to create the skipgram loss for edge structure

    Parameters
    ----------
    embeddings: the embeddings features for the current subgraph.
    features: features from tensorflow dataset (not used)
    edge_list: edge list of the subgraph
    weights: weights of the edges in the subgraph
    params: other parameters

    Returns
    -------
    a tensor representing the edge prediction loss.
    """
    with tf.name_scope('edge_list_logits'):
        pairwise_inner_prods = tf.matmul(embeddings, embeddings, transpose_b=True,
                                         name='all_edges_logit')

        if len(edge_list.shape) == 2:
            edge_list = tf.expand_dims(edge_list, axis=0)
            pairwise_inner_prods = tf.expand_dims(pairwise_inner_prods, axis=0)
            no_batch = True
        else:
            no_batch = False

        edge_list_shape = tf.shape(edge_list)
        batch_size = edge_list.shape[0].value if edge_list.shape[0].value is not None else edge_list_shape[0]
        num_edges = edge_list.shape[1].value if edge_list.shape[1].value is not None else edge_list_shape[1]

        batch_index = tf.tile(
            tf.expand_dims(tf.expand_dims(tf.range(batch_size), -1), -1),
            tf.stack([1, num_edges, 1]))

        edge_index = tf.concat([batch_index, edge_list], axis=-1)
        edge_logit = tf.gather_nd(pairwise_inner_prods, edge_index)

        if no_batch:
            edge_logit = tf.squeeze(edge_logit, axis=0)

        return edge_logit


def _get_edge_loss(embeddings, features, params):
    # subgraph structure
    edge_list = features['edge_list']
    weights = features['weights']  # should be {0., 1.}
    if weights.shape[-1].value == 1:
        weights = tf.squeeze(weights, axis=-1)

    n_vert = tf.shape(features['vertex_index'])

    # Edge predictions
    edge_logits = _make_edge_logits(embeddings, features, edge_list, weights, params)

    # edge loss
    with tf.name_scope('edge_loss', values=[edge_logits, edge_list, weights]):
        skipgram_loss_fn = make_simple_skipgram_loss(12)
        edge_pred_loss = skipgram_loss_fn(edge_logits, n_vert, edge_list, weights, params)

        # normalize to get the per-edge loss for easy logging
        edge_pred_size = tf.shape(edge_logits)[-1]
        edge_pred_loss_normalized = tf.divide(edge_pred_loss, tf.to_float(edge_pred_size))

    return edge_pred_loss, edge_pred_loss_normalized


def _rerm_optimizer(loss, params, embedding_optimizer, global_optimizer):
    embedding_vars = [v for v in tf.trainable_variables() if "embedding" in v.name]

    global_vars = [v for v in tf.trainable_variables() if "embedding" not in v.name]
    global_step = tf.train.get_or_create_global_step()

    update_global_step = tf.assign_add(global_step, 1, name="global_step_update")

    embedding_optimizer_value = _get_value(embedding_optimizer)
    global_optimizer_value = _get_value(global_optimizer)

    if len(embedding_vars) > 0:
        embedding_update = embedding_optimizer_value.minimize(
            loss, var_list=embedding_vars, global_step=None)
    else:
        embedding_update = tf.identity(0.)  # meaningless

    if len(global_vars) > 0:

        global_vars = tf.trainable_variables("dragon")

        # gradients = global_optimizer_value.compute_gradients(loss, var_list=global_vars)
        # gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=1.0)
        # global_update = global_optimizer_value.apply_gradients(zip(gradients, global_vars),
        #                                                        global_step=None)

        global_update = global_optimizer_value.minimize(
            loss, var_list=global_vars, global_step=None)
    else:
        global_update = tf.identity(0.)

    with tf.control_dependencies([update_global_step]):
        basic_train_op = tf.group(embedding_update, global_update)

    train_op = basic_train_op

    return train_op


def treatment_response_model_fn_builder(label_task_weight=0.01,
                                        init_checkpoint=None,
                                        label_pred=True,
                                        unsupervised=False,
                                        global_optimizer=None,
                                        embedding_optimizer=None,
                                        regularization=None,
                                        treatment_cat=True,
                                        outcome_cat=True,
                                        polyak_train=False,
                                        polyak_restore=False):
    """
    Main use case: binary treatment and binary outcome.
    Non-categorical outcomes and treatments supported, but ¯\_(ツ)_/¯

    Parameters
    ----------
    label_task_weight
    init_checkpoint
    label_pred
    unsupervised
    global_optimizer
    embedding_optimizer
    regularization
    t_cat
    o_cat
    polyak_train

    Returns
    -------

    """

    """Returns `model_fn` closure for tf.Estimator."""
    if embedding_optimizer is None:
        embedding_optimizer = _default_embedding_optimizer
    if global_optimizer is None:
        global_optimizer = _default_global_optimizer


    def model_fn(features, labels, mode, params):

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        treatment = features['treatment']
        outcome = features['outcome']

        in_train = features['in_train']
        in_test = features['in_test']
        in_dev = features['in_dev']

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        # graph specific
        vertex_index = features['vertex_index']
        all_embeddings = _make_embedding_variable(params)

        vertex_embedding_shape = tf.concat(
            [tf.shape(vertex_index), [params['embedding_dim']]], axis=0,
            name='vertex_embedding_shape')

        # We flatten the vertex index prior to extracting embeddings
        # to maintain compatibility with the input columns.
        embeddings = tf.nn.embedding_lookup(all_embeddings, tf.reshape(vertex_index, [-1]))
        embeddings = tf.reshape(embeddings, vertex_embedding_shape, name='vertex_embeddings_batch')

        if label_pred and not unsupervised:
            print("label_pred and not unsupervised")
            label_loss, outcome_st_treat, outcome_st_no_treat, treat = \
                _create_or_get_dragonnet_rerm(embeddings, is_training, treatment, outcome, in_train,
                                              treatment_cat=treatment_cat, outcome_cat=outcome_cat)
            total_loss = label_loss

        elif unsupervised and not label_pred:
            print("unsupervised")
            edge_loss, edge_loss_normalized = _get_edge_loss(embeddings, features, params)

            tf.summary.scalar('edge_loss_normalized', edge_loss_normalized, family='loss')

            total_loss = edge_loss

        elif unsupervised and label_pred:
            print("label_pred and unsupervised")
            edge_loss, edge_loss_normalized = _get_edge_loss(embeddings, features, params)

            label_loss, outcome_st_treat, outcome_st_no_treat, treat = \
                _create_or_get_dragonnet_rerm(embeddings, is_training, treatment, outcome, in_train,
                                              treatment_cat=treatment_cat, outcome_cat=outcome_cat)

            tf.losses.add_loss(edge_loss)
            tf.losses.add_loss(label_task_weight*label_loss)

            tf.summary.scalar('edge_loss_normalized', edge_loss_normalized, family='loss')
            tf.summary.scalar('edge_loss', edge_loss, family='loss')
            tf.summary.scalar('label_loss', label_loss, family='loss')

            total_loss = edge_loss + label_task_weight*label_loss

        else:
            raise ValueError('At least one of unsupervised or label_pred must be true')

        reg_loss = tf.losses.get_regularization_loss()
        total_loss = total_loss + reg_loss

        # some extras
        if label_pred:
            if polyak_train:
                global_variables = tf.get_collection('trainable_variables', 'dragon_net')  # Get list of the global variables
                ema = tf.train.ExponentialMovingAverage(decay=0.95)
                ema_op = ema.apply(global_variables)

                # polyak_getter = _get_getter(ema)
                #
                # label_loss, outcome_st_treat, outcome_st_no_treat, treat = \
                #     _create_or_get_dragonnet_rerm(embeddings, is_training, treatment, outcome, in_train,
                #                                   treatment_cat=treatment_cat, outcome_cat=outcome_cat,
                #                                   getter=polyak_getter)


            # outcome given treatment
            treatment_float = tf.cast(treatment, tf.float32)

            make_label_prediction_summaries(outcome_st_treat['per_example_loss'],
                                            outcome_st_treat['logits'],
                                            outcome,
                                            in_train*treatment_float,
                                            "train-" + 'outcome_st_treat',
                                            is_categorical=outcome_cat)

            make_label_prediction_summaries(outcome_st_treat['per_example_loss'],
                                            outcome_st_treat['logits'],
                                            outcome,
                                            in_dev*treatment_float,
                                            "dev-" + 'outcome_st_treat',
                                            is_categorical=outcome_cat)

            # outcome given no treatment
            make_label_prediction_summaries(outcome_st_no_treat['per_example_loss'],
                                            outcome_st_no_treat['logits'],
                                            outcome,
                                            in_train*(1.-treatment_float),
                                            "train-" + 'outcome_st_no_treat',
                                            is_categorical=outcome_cat)

            make_label_prediction_summaries(outcome_st_no_treat['per_example_loss'],
                                            outcome_st_no_treat['logits'],
                                            outcome,
                                            in_dev*(1.-treatment_float),
                                            "dev-" + 'outcome_st_no_treat',
                                            is_categorical=outcome_cat)

            # treatment
            make_label_prediction_summaries(treat['per_example_loss'],
                                            treat['logits'],
                                            treatment,
                                            in_train,
                                            "train-" + 'treat',
                                            is_categorical=treatment_cat)

            make_label_prediction_summaries(treat['per_example_loss'],
                                            treat['logits'],
                                            treatment,
                                            in_dev,
                                            "dev-" + 'treat',
                                            is_categorical=treatment_cat)

        # load pre-trained model
        if init_checkpoint:
            tvars = tf.trainable_variables()
            name_to_variable = None
            (assignment_map, initialized_variable_names) = \
                get_assignment_map_from_checkpoint(tvars, init_checkpoint, name_to_variable)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

            if polyak_restore:
                restore_variables = ema.variables_to_restore()
                init_vars = tf.train.list_variables(init_checkpoint)
                init_vars_name = [n for n, v in init_vars]
                restore_variables = {k: v for k,v in restore_variables.items() if k in init_vars_name}
                print("RESTORE VARIABLES")
                print(restore_variables)

                tf.train.init_from_checkpoint(init_checkpoint, restore_variables)



        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        if mode == tf.estimator.ModeKeys.TRAIN:

            if polyak_train and label_pred:
                with tf.control_dependencies([ema_op]):
                    train_op = _rerm_optimizer(total_loss, params, embedding_optimizer, global_optimizer)
            else:
                train_op = _rerm_optimizer(total_loss, params, embedding_optimizer, global_optimizer)

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
            )

        elif mode == tf.estimator.ModeKeys.EVAL:

            # reload the model to get the loss computation right... it doesn't affect anything else
            test_loss, outcome_st_treat, outcome_st_no_treat, treat = \
                _create_or_get_dragonnet_rerm(embeddings, is_training, treatment, outcome, in_test,
                                              treatment_cat=treatment_cat, outcome_cat=outcome_cat)


            eval_feed = {'outcome': outcome, 'treatment': treatment, 'in_test': in_test}
            eval_feed.update({'per_example_loss_ot1': outcome_st_treat['per_example_loss'],
                              'logits_ot1': outcome_st_treat['logits']})
            eval_feed.update({'per_example_loss_ot0': outcome_st_no_treat['per_example_loss'],
                              'logits_ot0': outcome_st_no_treat['logits']})
            eval_feed.update({'per_example_loss_t': treat['per_example_loss'],
                              'logits_t': treat['logits']})

            # eval_metrics = (_treatment_outcome_eval_metric_fn, eval_feed)

            treatment_outcome_eval_metric_fn = _make_treatment_outcome_eval_metric_fn(treatment_cat, outcome_cat)
            eval_metrics = treatment_outcome_eval_metric_fn(**eval_feed)

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=test_loss,
                eval_metric_ops=eval_metrics)

        else:
            # remark: standard tensorflow workflow would be to only pass in testing data.
            # We're anticipating that all data may get passed in (due to possible relational structure)
            predictions = {'in_test': in_test[:, 0]}
            predictions.update({'treatment_probability': treat['expectations']})
            predictions.update({'expected_outcome_st_treatment': outcome_st_treat['expectations']})
            predictions.update({'expected_outcome_st_no_treatment': outcome_st_no_treat['expectations']})

            # need this info downstream... might as well save it here
            predictions.update({'outcome': outcome[:, 0]})
            predictions.update({'treatment': treatment[:, 0]})
            predictions.update({'vertex_index': vertex_index[:, 0]})

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode, predictions=predictions)

        return output_spec

    return model_fn
