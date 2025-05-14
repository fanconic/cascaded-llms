#!/bin/bash

OVERRIDES="results_dir=results_3_8_models base_model=Qwen/Qwen2.5-3B-Instruct device=cuda:0 precomputed.verification=false  precomputed.uncertainty=false precomputed.path=/home/azureuser/caf83/helm/results_3_8_models/run_20250510_224123_first_qwen_arc_easy/results_first_qwen_arc_easy_self_verification_1_large.csv"
python main.py --config-path=configs/ai2arc_easy --config-name=first_qwen $OVERRIDES

OVERRIDES="results_dir=results_3_8_models base_model=Qwen/Qwen2.5-3B-Instruct device=cuda:0 precomputed.verification=false  precomputed.uncertainty=false precomputed.path=/home/azureuser/caf83/helm/results_3_8_models/run_20250511_023835_first_qwen_arc_challenge/results_first_qwen_arc_challenge_self_verification_1_large.csv"
python main.py --config-path=configs/ai2arc_challenge --config-name=first_qwen $OVERRIDES
