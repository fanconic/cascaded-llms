#!/bin/bash

OVERRIDES="results_dir=results_3_8_models base_model=meta-llama/Llama-3.2-3B-Instruct device=cuda:3 precomputed.uncertainty=false precomputed.path=/home/azureuser/caf83/helm/results_3_8_models/run_20250510_224043_first_llama_arc_easy/results_first_llama_arc_easy_self_verification_1_large.csv"
python main.py --config-path=configs/ai2arc_easy --config-name=first_llama $OVERRIDES

OVERRIDES="results_dir=results_3_8_models base_model=meta-llama/Llama-3.2-3B-Instruct device=cuda:3 precomputed.uncertainty=false precomputed.path=/home/azureuser/caf83/helm/results_3_8_models/run_20250511_033108_first_llama_arc_challenge/results_first_llama_arc_challenge_self_verification_1_large.csv"
python main.py --config-path=configs/ai2arc_challenge --config-name=first_llama $OVERRIDES
