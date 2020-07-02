"""
A set of helper functions to reproduce all of the tables in the paper.
The inputs to this program are the outputs of running the rerm_model code with all possible
simulation settings, and data splits
"""

import os
import glob

import numpy as np
import pandas as pd

from semi_parametric_estimation.ate import ate_estimates


def ground_truth_and_naive_from_sim_log(sim_log):
    simulated = np.load(sim_log)
    y_0 = simulated['y_0']
    y_1 = simulated['y_1']
    t = simulated['treatments']

    ground_truth = y_1.mean() - y_0.mean()
    naive = y_1[t==1].mean() - y_0[t==1].mean()

    return ground_truth, naive


def _make_descale(sim_log):
    if sim_log is None:
        return lambda outcome: outcome

    simulated = np.load(sim_log)
    y = simulated['outcomes']

    # I idiotically scaled y for training without unscaling it at prediction time
    s = y.std()
    m = y.mean()

    def descale(outcome):
        return outcome*s + m

    return descale


def ate_from_rerm_tsv(tsv_path):
    """
    Takes the predicted expected outcomes and propensity scores for each vertex and computes
    the corresponding average treatment effect on the graph, using the test data

    Parameters
    ----------
    tsv_path path to tsv file output by RERM model
    sim_log path to simulation log file if the data was simulated
        NOTE: sim_log is only necessary because I (idiotically) normalized y for training but neglected to
         denormalize it at prediction time

    Returns
    -------

    """
    output = pd.read_csv(tsv_path, '\t')
    # output = convert_str_columns_to_float(output)

    y = output['outcome'].values
    t = output['treatment'].values
    q_t0 = output['expected_outcome_st_no_treatment'].values
    q_t1 = output['expected_outcome_st_treatment'].values
    g = output['treatment_probability'].values
    in_test = output['in_test'].values == 1
    in_train = np.logical_not(in_test)

    q_t0_test = q_t0[in_test]
    q_t1_test = q_t1[in_test]
    g_test = g[in_test]
    t_test = t[in_test]
    y_test = y[in_test]

    # bonus_estimates = ates_from_atts(q_t0_test, q_t1_test, g_test, t_test, y_test, truncate_level=0.03)
    # bonuss_estimates = {}
    # for k in bonus_estimates:
    #     bonuss_estimates[k + '_bonus'] = bonus_estimates[k]
    # all_ate_estimates.update(bonuss_estimates)

    # all_ate_estimates = ate_estimates(q_t0, q_t1, g, t, y, truncate_level=0.03)
    # all_ate_estimates = ate_estimates(q_t0[in_train], q_t1[in_train], g[in_train], t[in_train], y[in_train],
    #                               truncate_level=0.03)
    all_estimates = ate_estimates(q_t0_test, q_t1_test, g_test, t_test, y_test, truncate_level=0.03)

    # print(tsv_path)
    # print(all_ate_estimates)

    return all_estimates


def rerm_psi(output_dir, sim_log):
    """
    Expects that the data was split into k folds, and the predictions from each fold
    was saved in [output_dir]/[fold_identifier]/[output_name].
    (matches {}/*/*.tsv'.format(output_dir))

    :param output_dir:
    :return:
    """

    data_files = sorted(glob.glob('{}/*/*.tsv'.format(output_dir)))
    data_files += sorted(glob.glob('{}/*/predict/*.tsv'.format(output_dir), recursive=True))
    estimates = []
    for data_file in data_files:
        try:
            all_estimates = ate_from_rerm_tsv(data_file)
            # print(psi_estimates)
            estimates += [all_estimates]
        except:
            print('wtf')
            print(data_file)

    print(output_dir)

    avg_estimates = {}
    for k in all_estimates.keys():
        k_estimates = []
        for estimate in estimates:
            k_estimates += [estimate[k]]

        avg_estimates[k] = np.mean(k_estimates)
        avg_estimates[(k, 'std')] = np.std(k_estimates)

    return avg_estimates


def process_covariate_experiment(
        base_dir='~/networks/sim_from_covariate'):
    # for setting in ['simple', 'multiplicative', 'interaction']:
    outputs = []
    for covariate in ['age', 'region', 'registration']:
        covariate_dir = os.path.join(base_dir, "covariate" + covariate)
        for beta1 in [1.0, 10.0, 100.0]:
            print("beta1: {}".format(beta1))
            output_dir = os.path.join(covariate_dir, 'beta1{}'.format(beta1))
            sim_log = os.path.join(output_dir, 'seed0', 'simulated_data.npz')
            estimates = rerm_psi(output_dir, sim_log)

            # ground_truth, very_naive = ground_truth_and_naive_from_sim_log(sim_log)

            setting_info = {'covariate': covariate, 'beta1': beta1}
            outputs += [{**setting_info, **estimates}]

    return outputs


def process_exogeneity_experiment(
        base_dir='~/networks/sim_exogeneity/beta110.0'):
    # for setting in ['simple', 'multiplicative', 'interaction']:
    outputs = []
    for exog in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        print("exog: {}".format(exog))
        output_dir = os.path.join(base_dir, 'EXOG{}'.format(exog))
        sim_log = os.path.join(output_dir, 'seed0', 'simulated_data.npz')
        estimates = rerm_psi(output_dir, sim_log)

        # ground_truth, very_naive = ground_truth_and_naive_from_sim_log(sim_log)

        setting_info = {'covariate': 'region', 'exog': exog}
        outputs += [{**setting_info, **estimates}]

    return outputs


def main():

    print("*******************************************")
    print("Main Covariate-Based Simulation Experiment")
    print("*******************************************")

    ate_estimates = pd.DataFrame(process_covariate_experiment())
    # print(ate_estimates.keys())
    print(ate_estimates)
    print(ate_estimates[['covariate', 'beta1', 'very_naive', 'q_only', 'iptw', 'tmle', 'aiptw']])
    print(ate_estimates.loc[:, [('very_naive', 'std'), ('q_only', 'std'), ('iptw', 'std'), ('tmle', 'std'), ('aiptw', 'std')]])

    print("*******************************************")
    print("Two-stage Covariate-Based Simulation Experiment (Baseline)")
    print("*******************************************")

    ate_estimates = pd.DataFrame(process_covariate_experiment(
        base_dir='~/networks/two_stage_sim_from_covariate'))
    print(ate_estimates[['covariate', 'beta1', 'very_naive', 'q_only', 'iptw', 'tmle', 'aiptw']])
    print(ate_estimates.loc[:, [('very_naive', 'std'), ('q_only', 'std'), ('tmle', 'std'), ('aiptw', 'std')]])

    # ate_estimates = pd.DataFrame(process_exogeneity_experiment())
    # print(ate_estimates)
    # print(ate_estimates[['covariate', 'exog', 'very_naive', 'q_only', 'iptw', 'tmle', 'aiptw']])
    # print(ate_estimates.loc[:, [('very_naive', 'std'), ('q_only', 'std'), ('tmle', 'std')]])


if __name__ == '__main__':
    main()
    # pass
