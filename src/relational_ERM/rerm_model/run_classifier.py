# TODO: add switcher for simulation, and code for 2-stage experiment

import os
import argparse

import tensorflow as tf
import numpy as np
import pandas as pd

import relational_ERM.sampling.factories as factories
from relational_ERM.dataset.dataset import make_input_fn, get_dataset_fn, make_no_graph_input_fn
from relational_ERM.rerm_model.treatment_outcome_predictor import treatment_response_model_fn_builder
from relational_ERM.data_cleaning.pokec import load_data_pokec, process_pokec_attributes
from relational_ERM.data_cleaning.simulate_treatment_outcome import simulate_from_pokec_covariate, \
    simulate_exogeneity_experiment


def _str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def add_parser_model_arguments(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--init_checkpoint', type=str, default=None)

    parser.add_argument('--treatment', type=str, default='old_school')

    parser.add_argument('--do_train', default=False, action='store_true')
    parser.add_argument('--do_eval', default=False, action='store_true')
    parser.add_argument('--do_predict', default=False, action='store_true')

    # simulation parameters
    parser.add_argument('--simulated', type=str, default="attribute",
                        help="whether to use attribute ('attribute') or propensity based ('propensity') simulation")
    parser.add_argument('--covariate', type=str, default="region")
    parser.add_argument('--beta1', type=float, default=1., help="amount of confounding")
    parser.add_argument('--exogeneity', type=float, default=0.,
                        help="exogeneous confounding, used for propensity simulation")
    parser.add_argument("--base_propensities_path", type=str, default='',
                        help="path to .tsv file containing a 'propensity score' for each unit, "
                               "used for propensity score-based simulation")

    parser.add_argument('--unsupervised', default=False, action='store_true')
    parser.add_argument('--label_pred', default=False, action='store_true')

    # training aprameters
    parser.add_argument('--num_train_steps', type=int, default=50000)
    parser.add_argument('--batch-size', type=int, default=1, help='minibatch size')
    parser.add_argument('--embedding_learning_rate', type=float, default=0.025,
                        help='sgd learning rate for embedding updates')
    parser.add_argument('--global_learning_rate', type=float, default=1.,
                        help='sgd learning rate for global updates')
    parser.add_argument('--num_warmup_steps', type=int, default=100, help='Number of warmup steps')


    # model parameters
    parser.add_argument('--label_task_weight', type=float, default=1e-2,
                        help='weight to assign to label task.')
    # default set to match Node2Vec
    parser.add_argument('--embedding_dim', type=int, default=128,
                        help='Number of dimensions. Default is 128.')
    parser.add_argument('--embedding_trainable', type=str, default='true',
                        help='if false, the embeddings are frozen')
    parser.add_argument('--global_regularization', type=float, default=1.,
                        help='regularization scale for global variables')


    # tensorflow details
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--save_checkpoints_steps', type=int, default=1000)
    parser.add_argument("--keep_checkpoints", type=int, default=1)

    parser.add_argument('--use-xla', action='store_true', help='use XLA JIT compilation')


    # dataset arguments
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--proportion-censored', type=float, default=0.5,
                        help='proportion of censored vertex labels at train time.')

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


def _adjust_learning_rate(learning_rate, batch_size):
    if batch_size is not None:
        return learning_rate * batch_size

    return learning_rate


def _adjust_regularization(regularization, batch_size):
    if batch_size is not None:
        return regularization / batch_size

    return regularization


def _linear_warmup(init_lr, num_warmup_steps):
    """
        Implements linear warmup. I.e., if global_step < num_warmup_steps, the
        learning rate will be `global_step/num_warmup_steps * init_lr`.
    """

    global_step = tf.train.get_or_create_global_step()

    learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)

    global_steps_int = tf.cast(global_step, tf.int32)
    warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

    global_steps_float = tf.cast(global_steps_int, tf.float32)
    warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

    warmup_percent_done = global_steps_float / warmup_steps_float
    warmup_learning_rate = init_lr * warmup_percent_done

    is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
    learning_rate = (
        (1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)

    return learning_rate


def _make_global_optimizer(args):
    def fn():
        learning_rate = args.global_learning_rate
        learning_rate = _adjust_learning_rate(learning_rate, args.batch_size)
        learning_rate = _linear_warmup(learning_rate, args.num_warmup_steps)
        return tf.train.GradientDescentOptimizer(learning_rate)
    return fn


def _make_local_optimizer(args):
    def fn():
        learning_rate = args.embedding_learning_rate
        learning_rate = _adjust_learning_rate(learning_rate, args.batch_size)
        learning_rate = _linear_warmup(learning_rate, args.num_warmup_steps)
        return tf.train.GradientDescentOptimizer(learning_rate)
    return fn


def main():
    tf.enable_eager_execution()
    tf.logging.set_verbosity(tf.logging.INFO)

    parser = add_parser_model_arguments()
    args = parser.parse_args()

    print("load the data")
    graph_data, profiles = load_data_pokec(args.data_dir)

    print("Loaded data with {} vertices and {} edges".format(graph_data.num_vertices, graph_data.edge_list.shape[0]))

    np.random.seed(42) # use consistent seed for simulation
    if args.simulated == 'attribute':
        treatments, outcomes, y_0, y_1, t_prob = \
            simulate_from_pokec_covariate(args.data_dir,
                                          covariate=args.covariate,
                                          beta0=1.0,
                                          beta1=args.beta1,
                                          gamma=1.0)
    elif args.simulated == 'propensity':
        output = pd.read_csv(args.base_propensities_path, '\t')
        base_propensity_scores = output['treatment_probability'].values

        treatments, outcomes, y_0, y_1, t_prob= \
            simulate_exogeneity_experiment(base_propensity_scores,
                                           exogeneous_con=args.exogeneity,
                                            beta0=1.0,
                                            beta1=args.beta1,
                                            gamma=1.0)

    #  but let it change for data splitting and initialization
    tf.set_random_seed(args.seed)
    np.random.seed(args.seed+42)

    os.makedirs(args.output_dir, exist_ok=True)
    np.savez(os.path.join(args.output_dir, 'simulated_data'),
             treatments=treatments, outcomes=outcomes, y_0=y_0, y_1=y_1, t_prob=t_prob)

    treatment_cat = True
    outcome_cat = not outcomes.dtype == np.float32

    if not outcome_cat:
        # rescale outcome to reduce the sensitivity of training to optimization parameters
        outcomes = (outcomes - outcomes.mean()) / outcomes.std()

    if not args.do_train and not args.do_eval and not args.do_predict:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    tf.gfile.MakeDirs(args.output_dir)

    session_config = tf.ConfigProto(intra_op_parallelism_threads=0, inter_op_parallelism_threads=4)

    if args.use_xla:
        session_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

    run_config = tf.estimator.RunConfig(log_step_count_steps=10,
                                        model_dir=args.output_dir,
                                        save_checkpoints_steps=args.save_checkpoints_steps,
                                        keep_checkpoint_max=args.keep_checkpoints,
                                        # save_checkpoints_steps=None,
                                        # save_checkpoints_secs=None,
                                        save_summary_steps=10,
                                        session_config=session_config)

    # estimator setup

    num_train_steps = args.num_train_steps
    vertex_embedding_params = {
        'embedding_dim': args.embedding_dim,
        'embedding_trainable': _str2bool(args.embedding_trainable)
    }

    model_fn = treatment_response_model_fn_builder(
        label_task_weight=args.label_task_weight,
        init_checkpoint=args.init_checkpoint,
        label_pred=args.label_pred,
        unsupervised=args.unsupervised,
        global_optimizer=_make_global_optimizer(args),
        embedding_optimizer=_make_local_optimizer(args),
        regularization=None,
        treatment_cat=treatment_cat,
        outcome_cat=outcome_cat,
        polyak_train=True)

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        params={
            **vertex_embedding_params,
            'num_vertices': graph_data.num_vertices,
            'batch_size': args.batch_size
        },
        model_dir=args.output_dir,
        config=run_config)

    if args.do_train:
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Batch size = %d", args.batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)

        # subsample and process the data
        with tf.name_scope("training_data"):
            dataset_fn_train = get_dataset_fn(args.sampler, args)
            train_input_fn = make_input_fn(graph_data, args, treatments, outcomes, dataset_fn_train)

        # additional logging
        hooks = [tf.train.LoggingTensorHook({
            'loss': 'loss'
        },
            every_n_iter=100)]
        if args.label_pred:
            hooks += [
                tf.train.LoggingTensorHook({
                    # 'token_ids': 'token_ids',
                    # 'token_mask': 'token_mask',
                    # 'label_ids': 'label_ids',
                    # 'pred_in': 'summary/in_split/predictions',
                    # 'pred_out': 'summary/out_split/predictions',
                    # 'ra_in': 'summary/in_split/labels/kappa/batch_random_agreement/random_agreement',
                    # 'ra_out': 'summary/out_split/labels/kappa/batch_random_agreement/random_agreement',
                },
                    every_n_iter=1000)
            ]

        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if args.do_train and (args.do_eval or args.do_predict):
        # reload the model to get rid of unsupervised parts of the model
        trained_model_checkpoint = tf.train.latest_checkpoint(args.output_dir)
        model_fn = treatment_response_model_fn_builder(
            label_task_weight=args.label_task_weight,
            init_checkpoint=trained_model_checkpoint,
            label_pred=True,
            unsupervised=False,
            treatment_cat=treatment_cat,
            outcome_cat=outcome_cat,
            polyak_train=False,
            polyak_restore=False)

        estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            params={
                **vertex_embedding_params,
                'num_vertices': graph_data.num_vertices,
                'batch_size': args.batch_size
            },
            model_dir=args.output_dir,
            config=run_config)

    if args.do_eval:

        tf.logging.info("***** Running evaluation *****")
        # tf.logging.info("  Num examples = %d", len(eval_examples))
        tf.logging.info("  Batch size = %d", args.batch_size)

        # This tells the estimator to run through the entire set.
        eval_steps = None

        with tf.name_scope("evaluation_data"):
            eval_input_fn = make_no_graph_input_fn(graph_data, args, treatments, outcomes, filter_test=True)

        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    if args.do_predict:
        tf.logging.info("***** Running prediction*****")

        if not outcome_cat:
            # undo the normalization of the outputs
            m = outcomes.mean()
            s = outcomes.std()

            def descale(prediction):
                prediction['outcome'] = prediction['outcome'] * s + m
                prediction['expected_outcome_st_treatment'] = prediction['expected_outcome_st_treatment'] * s + m
                prediction['expected_outcome_st_no_treatment'] = prediction['expected_outcome_st_no_treatment'] * s + m
                return prediction
        else:
            # categorical Y wasn't rescaled, so no need to do this
            def descale(prediction):
                return prediction

        with tf.name_scope("evaluation_data"):
            predict_input_fn = make_no_graph_input_fn(graph_data, args, treatments, outcomes)

        result = estimator.predict(input_fn=predict_input_fn)

        output_predict_file = os.path.join(args.output_dir, "test_results.tsv")
        with tf.gfile.GFile(output_predict_file, "w") as writer:
            tf.logging.info("***** Predict results *****")

            attribute_names = ['vertex_index', 'in_test',
                               'treatment_probability',
                               'expected_outcome_st_treatment', 'expected_outcome_st_no_treatment',
                               'outcome', 'treatment']

            header = "\t".join(
                attribute_name for attribute_name in attribute_names) + "\n"
            writer.write(header)
            for prediction in result:
                prediction = descale(prediction)
                output_line = "\t".join(
                    str(prediction[attribute_name]) for attribute_name in attribute_names) + "\n"
                writer.write(output_line)


if __name__ == "__main__":
    # flags.mark_flag_as_required("input_files_or_glob")
    # flags.mark_flag_as_required("vocab_file")
    # flags.mark_flag_as_required("bert_config_file")
    # flags.mark_flag_as_required("output_dir")
    main()
