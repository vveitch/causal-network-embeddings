#!/bin/bash

INIT_DIR=../output/unsupervised_pokec_regional_embeddings/stable
INIT_FILE=$INIT_DIR/model.ckpt-1000
DATA_DIR=../dat/pokec/regional_subset
OUTPUT_DIR=../output/local_test/
PREDICT_FILE=~/networks/sim_from_covariate/covariateregion/beta110.0/seed0/test_results.tsv

rm -rf $OUTPUT_DIR

python -m relational_ERM.rerm_model.run_classifier \
  --seed 0 \
  --do_train \
  --do_eval \
  --do_predict \
  --data_dir=$DATA_DIR \
  --output_dir=$OUTPUT_DIR \
  --label_task_weight 5e-3\
  --num_train_steps 1000 \
  --batch-size 100 \
  --embedding_learning_rate 0.005 \
  --global_learning_rate 0.005 \
  --save_checkpoints_steps=500 \
  --label_pred \
  --unsupervised \
  --proportion-censored 0.1 \
  --simulated attribute \
  --embedding_trainable true \
  --beta1 1.0 \
  --covariate region
#  --init_checkpoint=$INIT_FILE \
#  --base_propensities_path=${PREDICT_FILE} \
#  --exogeneity=0.2