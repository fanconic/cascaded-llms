#!/bin/bash

OVERRIDE_RESULTS_DIR="results_dir=results_BLR batch_size=100"

python main.py --config-path=configs/ai2arc_easy --config-name=first_qwen $OVERRIDE_RESULTS_DIR
python main.py --config-path=configs/ai2arc_challenge --config-name=first_qwen $OVERRIDE_RESULTS_DIR
python main.py --config-path=configs/medqa --config-name=first_qwen $OVERRIDE_RESULTS_DIR
python main.py --config-path=configs/medmcqa --config-name=first_qwen $OVERRIDE_RESULTS_DIR

python main.py --config-path=configs/ai2arc_easy --config-name=first_llama $OVERRIDE_RESULTS_DIR
python main.py --config-path=configs/ai2arc_challenge --config-name=first_llama $OVERRIDE_RESULTS_DIR
python main.py --config-path=configs/medqa --config-name=first_llama $OVERRIDE_RESULTS_DIR
python main.py --config-path=configs/medmcqa --config-name=first_llama $OVERRIDE_RESULTS_DIR


