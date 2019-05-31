import tensorflow as tf
from scipy.special import expit
import numpy as np
import scipy.stats as stats
import os
from sklearn.linear_model import LogisticRegression, LinearRegression

from relational_ERM.data_cleaning.pokec import load_data_pokec, process_pokec_attributes


def logits_from_load_trained_embeddings(ckpt):
    # Say, '/tmp/model.ckpt' has the following tensors:
    #  -- name='old_scope_1/var1', shape=[20, 2]
    #  -- name='old_scope_1/var2', shape=[50, 4]
    #  -- name='old_scope_2/var3', shape=[100, 100]

    embeddings = tf.train.load_variable(ckpt, 'input_layer/vertex_index_embedding/embedding_weights')
    return np.sum(embeddings, 1)


def simulate_y(z, t, setting="A"):

    if setting == "A":
        # easy case

        mu_0 = z
        mu_1 = z + 4  # so the mean is 4 higher

        y_0 = np.random.normal(mu_0, 1.)
        y_1 = np.random.normal(mu_1, 1.)

    if setting == "B":
        # non-linear response

        noisy_z = np.random.normal(z, 2)

        mu_0 = 4*np.cos(noisy_z)
        mu_1 = 4*np.cos(noisy_z+3.14/2) + 4  # so the mean is 4 higher

        y_0 = np.random.normal(mu_0, 2)
        y_1 = np.random.normal(mu_1, 2)

    if setting == "C":
        # high complexity response

        noisy_z = np.random.normal(z, 2)

        # z_vec = np.expand_dims(noisy_z, 1)*np.ones([noisy_z.shape[0], 250])
        betas = np.random.uniform(-1, 1, 250)
        # new_z = np.sum(betas*z_vec, 1)
        new_z = np.sum(betas)*noisy_z

        mu_0 = new_z
        mu_1 = new_z + 4  # so the mean is 4 higher

        y_0 = np.random.normal(mu_0, 1)
        y_1 = np.random.normal(mu_1, 1)

    if setting == "D":
        # non-normal noise

        mu_0 = z
        mu_1 = z + 4  # so the mean is 4 higher

        y_0 = stats.t.rvs(2.5, loc=mu_0)
        y_1 = stats.t.rvs(2.5, loc=mu_1)

    if setting == "E":
        z_mod = (z-z.min()) / (z.max() - z.min())  # rescale to [0, 1]
        z_mod = (z_mod - 0.5) * 6. # rescale to [-3, 3]

        y0_prob = expit(z_mod)  # probabilities between 0.047 and 0.95 (enough to be interesting w/o being pathological)
        y1_prob = expit(z_mod + 0.5*t)
        y_0 = np.random.binomial(1, y0_prob)
        y_1 = np.random.binomial(1, y1_prob)

    y = np.where(t, y_1, y_0)

    if setting == "E":
        y = y.astype(np.int32)
    else:
        y = y.astype(np.float32)

    return y, y_1, y_0


def alt_simulate_from_continuous_confounder(z, t, setting="A"):

    std_z = (z-z.mean())/z.std()

    if setting=="A":
        noisy_z = np.random.normal(std_z, 2)

        mu_0 = 4*np.cos(noisy_z)
        mu_1 = 4*np.cos(noisy_z+3.14/2) + 4  # so the mean is 4 higher

        y_0 = np.random.normal(mu_0, 2)
        y_1 = np.random.normal(mu_1, 2)


    if setting=="B":
        noisy_z = np.random.normal(std_z, 2)

        z_vec = np.expand_dims(noisy_z,1)*np.ones([noisy_z.shape[0], 250])
        betas = np.random.uniform(-1, 1, 250)
        new_z = np.sum(betas*z_vec,1)

        mu_0 = new_z
        mu_1 = new_z + 4  # so the mean is 4 higher

        y_0 = np.random.normal(mu_0, 1)
        y_1 = np.random.normal(mu_1, 1)


    if setting=="C":
        noisy_z = np.random.normal(std_z, 2)

        mu_0 = 4*np.cos(noisy_z)
        mu_1 = 4*np.cos(noisy_z+3.14/2) + 4  # so the mean is 4 higher

        y_0 = mu_0 + np.random.gamma(2., 4., mu_0.shape[0])
        y_1 = mu_1 + np.random.gamma(2., 4., mu_0.shape[0])


    if setting=="D":
        noisy_z = np.random.normal(std_z, 2)

        z_vec = np.expand_dims(noisy_z, 1)*np.ones([noisy_z.shape[0], 250])
        betas = np.random.uniform(-1, 1, 250)
        new_z = np.sum(betas*z_vec,1)

        mu_0 = new_z
        mu_1 = new_z + 4  # so the mean is 4 higher

        y_0 = mu_0 + np.random.gamma(2., 4., mu_0.shape[0])
        y_1 = mu_1 + np.random.gamma(2., 4., mu_0.shape[0])


    if setting=="E":

        mu_0 = std_z
        mu_1 = std_z + 4  # so the mean is 4 higher

        y_0 = mu_0 + np.random.gamma(2., 4., mu_0.shape[0])
        y_1 = mu_1 + np.random.gamma(2., 4., mu_0.shape[0])


    if setting=="F":

        mu_0 = std_z
        mu_1 = std_z + 4  # so the mean is 4 higher

        y_0 = mu_0 + np.random.normal(0., 1., mu_0.shape[0])
        y_1 = mu_1 + np.random.normal(0., 1., mu_0.shape[0])


    y = np.where(t, y_1, y_0)
    y = y.astype(np.float32)

    return y, y_1, y_0

def simulate_from_continuous_confounder(z, setting="A"):

    if setting=="A":
        # y easy to estimate, t kinda annoying

        t_coeff = np.random.uniform(1, 3)

        std_z = (z - z.mean()) / z.std()
        t_logit_mean = t_coeff * std_z
        t_logit = np.random.normal(loc=t_logit_mean,
                                   scale=0.1)
        t_prob = expit(t_logit)
        t = np.random.binomial(1, t_prob)

        y_0 = np.random.normal(z, 1)
        y_1 = np.random.normal(4 + z, 1)

    if setting=="B":
        # everything kinda annoying to estimate
        t_coeff = np.random.uniform(1, 3)

        std_z = (z - z.mean()) / z.std()
        t_logit_mean = t_coeff * std_z
        t_logit = np.random.normal(loc=t_logit_mean,
                                   scale=0.1)
        t_prob = expit(t_logit)
        t = np.random.binomial(1, t_prob)

        mu_0 = np.exp(std_z)
        mu_1 = std_z + mu_0.mean() + 4  # so the mean is 4 higher
        y_0 = np.random.normal(mu_0, 1)
        y_1 = np.random.normal(mu_1, 1)

    if setting=="C":
        # t easy to estimate, y kinda annoying

        print("is this even working?")

        std_z = (z - z.mean()) / z.std()
        t_prob = expit(1.5*std_z)
        t = np.random.binomial(1, t_prob)

        mu_0 = np.exp(std_z) + np.sqrt(np.abs(z))
        mu_1 = z - z.mean() + mu_0.mean() + 4  # so the mean is 4 higher
        y_0 = np.random.normal(mu_0, 10)
        y_1 = np.random.normal(mu_1, 10)

    if setting=="D":
        # everying easy

        std_z = (z - z.mean()) / z.std()
        t_prob = expit(std_z)
        t = np.random.binomial(1, t_prob)

        mu_0 = np.exp(std_z)
        mu_1 = std_z + mu_0.mean() + 4  # so the mean is 4 higher
        y_0 = np.random.normal(mu_0, 1)
        y_1 = np.random.normal(mu_1, 1)


    y = np.where(t, y_1, y_0)
    t = t.astype(np.int32)

    return t, y, y_1, y_0, t_prob, z


def simulate_from_pokec_covariates0(data_dir, setting="A"):
    graph_data, profiles = load_data_pokec(data_dir)
    pokec_features = process_pokec_attributes(profiles)

    # predictable covariates
    covs = ['scaled_registration',
            'scaled_age',
            'region']
    # 'sign_in_zodiac',
    # 'relation_to_casual_sex']
    clean_age = np.where(np.isnan(pokec_features['scaled_age']),
                         np.zeros_like(pokec_features['scaled_age']), pokec_features['scaled_age'])
    pokec_features['scaled_age'] = clean_age
    region = pokec_features['region']
    pokec_features['region'] = (region - region.mean()) / region.std()

    cov_array = np.zeros([covs.__len__(), pokec_features['region'].shape[0]])
    for idx, cov in enumerate(covs):
        cov_array[idx] = pokec_features[cov]

    coeff = np.random.uniform(-2, 2, [covs.__len__(), 1])

    z = 0.5 + np.sum(coeff*cov_array, 0)
    t_prob = expit(z)
    t = np.random.binomial(1, t_prob)

    y, y_1, y_0 = simulate_y(z, t, setting)

    return t, y, y_0, y_1, t_prob, z


def simulate_from_pokec_covariates2(data_dir, setting="A"):
    """mimic TMLE experiment to see what happens"""
    graph_data, profiles = load_data_pokec(data_dir)
    pokec_features = process_pokec_attributes(profiles)

    region = pokec_features['region']
    region = np.searchsorted(np.unique(region), region) - 1.


    old_school = pokec_features['old_school']
    age = pokec_features['scaled_age']
    age_cat = np.where(age < 0, -1., 1.)
    age_cat[np.isnan(age)] = 0

    # simulate
    covs = [old_school, region, age_cat]

    # z = 0.
    # for cov in covs:
    #     z += np.random.uniform(-2, 2)*cov
    z = (2*(region < 1)-1)
    t_prob = expit(z)
    t = np.random.binomial(1, t_prob)

    zz = 0.
    for cov in covs:
        zz += np.random.uniform(-1, 3)*cov

    y, y_1, y_0 = simulate_y(zz, t, setting)

    return t, y, y_0, y_1, t_prob, z


def simulate_from_pokec_covariates(data_dir, setting="A", discretize_covariates=True, easy_t=True):
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

    if discretize_covariates:
        age = pokec_features['scaled_age']
        age_cat = np.where(age < 0, -1., 1.)
        age_cat[np.isnan(age)] = 0

        old_school = pokec_features['old_school'] # binarized version of registration

        # simulate
        covs = [old_school, region, age_cat]

    else:
        scaled_age = pokec_features['scaled_age']
        scaled_age[np.isnan(scaled_age)] = 0

        scaled_region = (region - region.mean()) / region.std()
        registration = pokec_features['scaled_registration']


        covs = [registration, scaled_region, scaled_age]

    # treatment
    if easy_t:
        z = (2 * (region < 1) - 1)
        t_prob = expit(z)
        t = np.random.binomial(1, t_prob)
    else:
        z = 0.
        for cov in covs:
            z += np.random.uniform(-1, 3) * cov

        z = (z-z.min()) / (z.max() - z.min())  # rescale to [0, 1]
        z = (z - 0.5) * 6. # rescale to [-3, 3]
        t_prob = expit(z)  # probabilities between 0.047 and 0.95 (enough to be interesting w/o being pathological)
        t = np.random.binomial(1, t_prob)

    # confounding
    zz = 0.
    for cov in covs:
        zz += np.random.uniform(-1, 3) * cov

    y, y_1, y_0 = simulate_y(zz, t, setting)

    return t, y, y_0, y_1, t_prob, z, zz


if __name__ == '__main__':
    # main()
    tf.enable_eager_execution()
    data_dir = '../dat/networks/pokec/regional_subset'
    graph_data, profiles = load_data_pokec(data_dir)
    pokec_features = process_pokec_attributes(profiles)

    sbm_embedding = np.loadtxt('../dat/networks/pokec/regional_subset/svinetk128groups.txt')
    sbm_embedding = sbm_embedding[:,1:] # drop the first column of embedding
    sbm_embedding = sbm_embedding[sbm_embedding[:,0].argsort()]
    sbm_embedding = sbm_embedding[:,1:]

    print("Loaded data with {} vertices and {} edges".format(graph_data.num_vertices, graph_data.edge_list.shape[0]))

    reps = 25

    for setting in ['A', 'B', 'C', 'D', 'E']:
        for discretize_covariates in [True, False]:
            for easy_t in [True, False]:
                mse = np.zeros([reps, 3])
                for itr in range(reps):
                    filename = "setting_"+setting+ \
                        "_discrete_"+str(discretize_covariates)+ \
                        "_easy_"+str(easy_t)+".csv"
                    print(filename)
                    treatments, outcomes, y_0, y_1, t_prob, z, zz = \
                        simulate_from_pokec_covariates(data_dir,
                                                       setting=setting,
                                                       discretize_covariates=discretize_covariates,
                                                       easy_t=easy_t
                                                       )

                    output_dir = '../../output/local_test/'
                    os.makedirs(output_dir, exist_ok=True)
                    np.savez(os.path.join(output_dir, 'simulated_data'),
                             treatments=treatments, outcomes=outcomes, y_0=y_0, y_1=y_1, t_prob=t_prob, t_latent=z, y_latent=zz)

                    treatment_cat = True
                    outcome_cat = not (outcomes.dtype == np.float32)
                    if not outcome_cat:
                        # rescale outcome to reduce the sensitivity of training to optimization parameters
                        outcomes = (outcomes - outcomes.mean()) / outcomes.std()

                    if setting == 'E':
                        true_trtcoeff = 0.5
                        logreg = LogisticRegression(C=1e16, solver='lbfgs')
                        X = np.column_stack([treatments, sbm_embedding])
                        Y = outcomes
                        logreg.fit(X, Y)
                        assert logreg.coef_.shape[1] == sbm_embedding.shape[1] + 1
                        est_t = logreg.coef_[0][0]
                        mse[itr] = np.array([est_t, true_trtcoeff, (est_t-true_trtcoeff)**2])
                    else:
                        linreg = LinearRegression()
                        X = np.column_stack([treatments, sbm_embedding])
                        Y = outcomes
                        linreg.fit(X, Y)
                        assert linreg.coef_.shape[0] == sbm_embedding.shape[1] + 1
                        est_t = linreg.coef_[0]
                        true_trtcoeff = np.mean(y_1-y_0)
                        mse[itr] = np.array([est_t, true_trtcoeff, (est_t-true_trtcoeff)**2])
                    print(mse[itr])
                np.savetxt(filename, mse, delimiter=",")
                print(np.mean(mse[:,2]), np.std(mse[:,2]))












