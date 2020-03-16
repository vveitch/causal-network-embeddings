#!/bin/bash

DATA_DIR=../dat/networks/pokec/regional_subset
OUTPUT_DIR=../output/unsupervised_pokec_regional_embeddings/stable/

rm -rf $OUTPUT_DIR

python -m relational_ERM.rerm_model.run_classifier \
      --do_train \
      --data_dir=${DATA_DIR} \
      --output_dir=${OUTPUT_DIR} \
      --batch-size 256 \
      --embedding_learning_rate=0.005 \
      --save_checkpoints_steps=1000 \
      --unsupervised