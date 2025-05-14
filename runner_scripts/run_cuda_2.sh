#!/bin/bash

OVERRIDES="results_dir=results_3_8_models base_model=meta-llama/Llama-3.2-3B-Instruct device=cuda:2  precomputed.uncertainty=false precomputed.path=/home/azureuser/caf83/helm/results_3_8_models/run_20250510_224047_first_llama_medmcqa/results_first_llama_medmcqa_self_verification_1_large.csv"
python main.py --config-path=configs/medmcqa --config-name=first_llama $OVERRIDES

OVERRIDES="results_dir=results_3_8_models base_model=meta-llama/Llama-3.2-3B-Instruct device=cuda:2  precomputed.uncertainty=false precomputed.path=/home/azureuser/caf83/helm/results_3_8_models/run_20250511_064733_first_llama_medqa/results_first_llama_medqa_self_verification_1_large.csv"
python main.py --config-path=configs/medqa --config-name=first_llama $OVERRIDES