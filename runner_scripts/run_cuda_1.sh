#!/bin/bash

OVERRIDES="results_dir=results_3_8_models base_model=Qwen/Qwen2.5-3B-Instruct device=cuda:1 precomputed.verification=false  precomputed.uncertainty=false precomputed.path=/home/azureuser/caf83/helm/results_3_8_models/run_20250510_224055_first_qwen_medmcqa/results_first_qwen_medmcqa_self_verification_1_large.csv"
python main.py --config-path=configs/medmcqa --config-name=first_qwen $OVERRIDES


OVERRIDES="results_dir=results_3_8_models base_model=Qwen/Qwen2.5-3B-Instruct device=cuda:1 precomputed.verification=false  precomputed.uncertainty=false precomputed.path=/home/azureuser/caf83/helm/results_3_8_models/run_20250511_074544_first_qwen_medqa/results_first_qwen_medqa_self_verification_1_large.csv"
python main.py --config-path=configs/medqa --config-name=first_qwen $OVERRIDES
