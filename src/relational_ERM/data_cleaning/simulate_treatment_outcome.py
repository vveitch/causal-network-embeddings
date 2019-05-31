import tensorflow as tf
from scipy.special import expit, logit
import numpy as np
import scipy.stats as stats

from relational_ERM.data_cleaning.pokec import load_data_pokec, process_pokec_attributes


def simulate_y(propensities, treatment, beta0=1.0, beta1=1.0, gamma=1.0):
    confounding = (propensities - 0.5).astype(np.float32)

    noise = np.random.normal(0.,1.,size=propensities.shape[0]).astype(np.float32)

    y0 = beta1 * confounding
    y1 = beta0 + y0
    y = (1. - treatment) * y0 + treatment * y1 + gamma * noise

    return y, y0, y1


def simulate_from_pokec_covariate(data_dir, covariate='region', beta0=1.0, beta1=1.0, gamma=1.0):
    graph_data, profiles = load_data_pokec(data_dir)
    pokec_features = process_pokec_attributes(profiles)

    # predictable covariates
    covs = ['scaled_registration',
            'scaled_age',
            'region']
    # 'sign_in_zodiac',
    # 'relation_to_casual_sex']

    # reindex region to 0, 1, 2
    region = pokec_features['region']
    region = np.searchsorted(np.unique(region), region) - 1.

    age = pokec_features['scaled_age']
    age_cat = np.where(age < 0., -1., 1.)
    age_cat[np.isnan(age)] = 0

    registration = pokec_features['scaled_registration']
    registration_cat = np.where(registration < -0.5, -1., 0.)
    registration_cat[registration > 0.5] = 1.

    if covariate == 'region':
        confounder = region
    elif covariate == 'age':
        confounder = age_cat
    elif covariate == 'registration':
        confounder = registration_cat
    else:
        raise Exception("covariate name not recognized")

    # simulate treatments and outcomes
    propensities = 0.5 + 0.35 * confounder
    treatment = np.random.binomial(1, propensities)
    y, y0, y1 = simulate_y(propensities, treatment, beta0=beta0, beta1=beta1, gamma=gamma)

    t = treatment.astype(np.int32)
    y = y.astype(np.float32)
    y0 = y0.astype(np.float32)
    y1 = y1.astype(np.float32)

    return t, y, y0, y1, propensities


def simulate_exogeneity_experiment(base_propensity_scores, exogeneous_con=0.,
                                   beta0=1.0, beta1=1.0, gamma=1.0):

    extra_confounding = np.random.normal(0, 1, base_propensity_scores.shape[0]).astype(np.float32)

    propensities = expit((1.-exogeneous_con)*logit(base_propensity_scores) +
                                  exogeneous_con * extra_confounding).astype(np.float32)

    treatment = np.random.binomial(1, propensities)
    y, y0, y1 = simulate_y(propensities, treatment, beta0=beta0, beta1=beta1, gamma=gamma)

    t = treatment.astype(np.int32)
    y = y.astype(np.float32)
    y0 = y0.astype(np.float32)
    y1 = y1.astype(np.float32)

    return t, y, y0, y1, propensities


def main():
    tf.enable_eager_execution()
    data_dir = '../../dat/networks/pokec/regional_subset'

    t, y, y0, y1, propensities = simulate_from_pokec_covariate(data_dir, covariate='region', beta0=1.0,
                                                               beta1=10.0, gamma=1.0)
    print(y[t==1].mean()-y[t==0].mean())


if __name__ == '__main__':
    main()