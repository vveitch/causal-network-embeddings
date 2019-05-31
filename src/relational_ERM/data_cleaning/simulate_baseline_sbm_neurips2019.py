# import tensorflow as tf
from scipy.special import expit
import numpy as np
import numpy.random as npr
import pandas as pd
import scipy.stats as stats
import os
from fnmatch import fnmatch
from sklearn.linear_model import LogisticRegression, LinearRegression

# from relational_ERM.data_cleaning.pokec import load_data_pokec, process_pokec_attributes

if __name__ == '__main__':
	# main()
	# tf.enable_eager_execution()
	# data_dir = '../dat/networks/pokec/regional_subset'
	# graph_data, profiles = load_data_pokec(data_dir)
	# pokec_features = process_pokec_attributes(profiles)

	data_dir = '/proj/sml/projects/causal-embeddings/rerm/pokec/sim_from_covariate/'

	sbm_embedding = np.loadtxt('/home/yixinwang/causal-spe-embeddings/dat/networks/pokec/regional_subset/svinetk128groups.txt')
	sbm_embedding = sbm_embedding[:,1:] # drop the first column of embedding
	sbm_embedding = sbm_embedding[sbm_embedding[:,0].argsort()]
	sbm_embedding = sbm_embedding[:,1:]

	reps = 25

	filenames = []


	root_dir = "/proj/sml/projects/causal-embeddings/rerm/pokec/sim_from_covariate"
	for folder in ['covariateregistration', 'covariateregion', 'covariateage']:
		root = os.path.join(root_dir, folder)

		pattern = '*simulated_data.npz'
		
		for path, subdirs, files in os.walk(root):
			for name in files:
				if fnmatch(name, pattern):
					filenames.append(os.path.join(path, name)) 
					print(filenames)

	ress = np.empty((0, 8))
	for file in filenames:
		print("filename", file)
		dat = np.load(file)
		treatments, outcomes, y_0, y_1, t_prob = \
			dat['treatments'], dat['outcomes'], dat['y_0'], dat['y_1'], dat['t_prob']
		mse = np.zeros([reps, 5])

		outfile = '_'.join(file.split('/')[-4:-1])+'_mse.csv'
		X = np.column_stack([treatments, sbm_embedding])
		Y = outcomes	
		n = X.shape[0]
		train_prop = 0.9
		for itr in range(reps):
			print("itr", itr)
			linreg = LinearRegression()
			train_idx = npr.choice(np.arange(n), int(train_prop * n), replace=False)

			X_tr, Y_tr, y_1_tr, y_0_tr = X[train_idx], Y[train_idx], y_1[train_idx], y_0[train_idx]

			linreg.fit(X_tr, Y_tr)
			assert linreg.coef_.shape[0] == sbm_embedding.shape[1] + 1
			est_t = linreg.coef_[0]
			true_trtcoeff = np.mean(y_1_tr-y_0_tr)
			mse[itr] = np.array([itr, est_t, true_trtcoeff, (est_t-true_trtcoeff)**2, abs(est_t-true_trtcoeff)])

		print(mse)
		res = np.column_stack([np.repeat(np.array(file.split('/')[-4:-1])[np.newaxis,:], reps, axis=0), mse])
		ress = np.row_stack([ress, res])

	ress = pd.DataFrame(ress, columns=['covariate', 'beta', 'seed', 'split_itr', 'est_ate', 'smp_ate', '(est_ate-smp_ate)^2', 'abs(est_ate-smp_ate)'])

	ress.to_csv("sbm_baseline_res.csv")
