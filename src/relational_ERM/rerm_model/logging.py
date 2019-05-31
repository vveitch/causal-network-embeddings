import tensorflow as tf


def batch_random_agreement(labels, predictions, weights, name=None):
    """ Computes the probability of random agreement between the
    labels and predictions assuming independence.

    Parameters
    ----------
    labels: a tensor of any shape taking values in {0, 1}.
    predictions: a tensor of the same shape as labels taking values in {0, 1}.
    weights: a tensor that can be broadcasted to labels.
    name: an optional name for the operation.

    Returns
    -------
    random_agreement: a scalar tensor representing the probability of random
        agreement.
    """
    with tf.name_scope(name, 'batch_random_agreement', [labels, predictions, weights]):
        weights_mean = tf.reduce_mean(weights)
        weights_mean = tf.where(tf.not_equal(weights_mean, 0), weights_mean, 1)

        labels = tf.to_float(labels)
        predictions = tf.to_float(predictions)

        p_labels = tf.metrics.mean(labels * weights / weights_mean)[1]
        p_predictions = tf.metrics.mean(predictions * weights / weights_mean)[1]

        random_agreement = tf.identity(
            p_labels * p_predictions + (1 - p_labels) * (1 - p_predictions),
            name='random_agreement')

        print(random_agreement.name)

    return random_agreement


def batch_kappa(labels, predictions, weights, name=None):
    """ Computes Cohen's kappa on the given batch of predictions.

    Parameters
    ----------
    labels: a tensor of any shape taking values in {0, 1}.
    predictions: a tensor of the same shape as labels taking values in {0, 1}.
    weights: a tensor that can be broadcasted to labels.
    name: an optional name for the operation.

    Returns
    -------
    kappa: a scalar tensor representing the Kappa measure of agreement
        between labels and predictions.
    """
    with tf.name_scope(name, 'batch_kappa', [labels, predictions, weights]):
        accuracy = tf.metrics.accuracy(labels, predictions, weights=weights)[1]
        random_agreement = batch_random_agreement(labels, predictions, weights)

        # hack for small batch sizes
        random_agreement = tf.clip_by_value(random_agreement, 0.001, 0.999)

        kappa = tf.divide(
            accuracy - random_agreement, 1 - random_agreement,
            name='kappa')

    return kappa


def make_label_prediction_summaries(per_example_loss, logits, labels, split, family, is_categorical=True):
    with tf.name_scope("summary"+"/"+family):
        if is_categorical:
            hard_predictions = tf.argmax(logits, axis=-1, output_type=tf.int32, name='hard_predictions')
            soft_predictions = tf.argmax(tf.nn.log_softmax(logits, axis=-1), axis=-1, output_type=tf.int32, name='soft_predictions')

            accuracy = tf.metrics.accuracy(labels, hard_predictions, weights=split, metrics_collections='labels')
            precision = tf.metrics.precision(labels, hard_predictions, weights=split, metrics_collections='labels')
            recall = tf.metrics.recall(labels, hard_predictions, weights=split, metrics_collections='labels')
            f1 = tf.contrib.metrics.f1_score(labels, soft_predictions, weights=split, metrics_collections='labels')
            kappa = batch_kappa(labels, hard_predictions, weights=split, name='labels/kappa')

            tf.summary.scalar('accuracy', accuracy[1], family=family)
            tf.summary.scalar('precision', precision[1], family=family)
            tf.summary.scalar('recall', recall[1], family=family)
            tf.summary.scalar('f1', f1[1], family=family)
            tf.summary.scalar('kappa', kappa, family=family)

        loss = tf.metrics.mean(per_example_loss, weights=split)
        # censored_per_example_loss = split * per_example_loss
        # loss = tf.reduce_sum(censored_per_example_loss) / tf.reduce_sum(split)

        tf.summary.scalar('loss', loss[1], family=family)


def label_eval_metric_fn(per_example_loss, label_ids, logits, split=None, family='', is_categorical=True):

    loss = tf.metrics.mean(per_example_loss, weights=split)
    metrics = {family+"/eval_loss": loss}

    if is_categorical:
        hard_predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        soft_predictions = tf.argmax(tf.nn.log_softmax(logits, axis=-1), axis=-1, output_type=tf.int32,
                                     name='soft_predictions')

        accuracy = tf.metrics.accuracy(label_ids, hard_predictions, weights=split)
        precision = tf.metrics.precision(label_ids, hard_predictions, weights=split)
        recall = tf.metrics.recall(label_ids, hard_predictions, weights=split)
        f1 = tf.contrib.metrics.f1_score(label_ids, soft_predictions, weights=split)
        # kappa = batch_kappa(label_ids, hard_predictions, weights=split, name='labels/kappa')

        metrics.update({
            family+"/eval_accuracy": accuracy,
            family+"/eval_precision": precision,
            family+"/eval_recall": recall,
            family+"/eval_f1": f1})

    return metrics


def unsupervised_eval_metric_fn(masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
                                masked_lm_weights):
    """Computes the loss and accuracy of the rerm_model."""
    masked_lm_log_probs = tf.reshape(masked_lm_log_probs,
                                     [-1, masked_lm_log_probs.shape[-1]])
    masked_lm_predictions = tf.argmax(
        masked_lm_log_probs, axis=-1, output_type=tf.int32)
    masked_lm_example_loss = tf.reshape(masked_lm_example_loss, [-1])
    masked_lm_ids = tf.reshape(masked_lm_ids, [-1])
    masked_lm_weights = tf.reshape(masked_lm_weights, [-1])
    masked_lm_accuracy = tf.metrics.accuracy(
        labels=masked_lm_ids,
        predictions=masked_lm_predictions,
        weights=masked_lm_weights)
    masked_lm_mean_loss = tf.metrics.mean(
        values=masked_lm_example_loss, weights=masked_lm_weights)

    return {
        "masked_lm_accuracy": masked_lm_accuracy,
        "masked_lm_loss": masked_lm_mean_loss,
    }