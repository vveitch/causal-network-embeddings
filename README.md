# Introduction

This repository contains software and data for "Using Embeddings to Correct for Unobserved Confounding in Networks
" (`arxiv:1902.04114`).
The paper describes a method to use network structured data to correct for unobserved confounding in causal inference.
For example, a social network carries information about the people in it.
We want to assess the effect of some treatment on people in the network using observational data.
In general, treatment assignment and outcome may be rely on latent attributes of the people.
We show how to use the network itself to correct for such unobserved confounding. 


This software builds on relational ERM https://github.com/wooden-spoon/relational-ERM. 
See that repository for detailed instructions on building predictive models using network embedding methods.


# Requirements
1. Python 3.6 with numpy and pandas
2. Tensorflow *1.11*
3. gcc


# Setup
Run the following command in src to build the graph sampler tensorflow ops:

python setup.py build_ext --inplace

# Data
We include pre-processed Pokec data for convenience. 
Raw data from https://snap.stanford.edu/data/soc-Pokec.html
We include the pre-processing scripts in the release; these can be modified as required.

# Reproducing the experiments
The default settings for the code match the settings used in the paper.
These match the default settings used by relational ERM (i.e., we didn't tune anything).

You'll run the code from `src` as 
`./relational_ERM/submit_scripts/run_model.sh`
Changing flags in this file will replicate different experiments.
The simulation setting is controlled by the `--simulated` flag. 
Options are attribute ('attribute') or propensity based ('propensity') simulation.
The later can be used to reproduce the exogeneity experiments.

To reproduce the two-stage training, run with `embedding_trainable=false`

Finally, the effect estimates can be reproduced by running `python -m effect_estimates.compute_ate`.
This takes in the predictions of the relational erm model (in tsv format) and passes them into downstream estimators
of the causal effect.

# Misc.
The experiments in the paper initialize from node embeddings that were pre-trained using a purely unsupervised objective.
To recreate the initialization embeddings, run `run_unsupervised.sh`. Then, uncomment `--init_checkpoint=$INIT_FILE` in `run_model.sh`

