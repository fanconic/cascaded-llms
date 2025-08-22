#!/bin/bash

OVERRIDE_RESULTS_DIR="results_dir=results_BLR_50 batch_size=100 calibration_size=50 device=cuda:0"

python main.py --config-path=configs/ai2arc_easy --config-name=first_qwen_calibrated $OVERRIDE_RESULTS_DIR
python main.py --config-path=configs/ai2arc_challenge --config-name=first_qwen_calibrated $OVERRIDE_RESULTS_DIR
python main.py --config-path=configs/medqa --config-name=first_qwen_calibrated $OVERRIDE_RESULTS_DIR
python main.py --config-path=configs/medmcqa --config-name=first_qwen_calibrated $OVERRIDE_RESULTS_DIR

python main.py --config-path=configs/ai2arc_easy --config-name=first_llama_calibrated $OVERRIDE_RESULTS_DIR
python main.py --config-path=configs/ai2arc_challenge --config-name=first_llama_calibrated $OVERRIDE_RESULTS_DIR
python main.py --config-path=configs/medqa --config-name=first_llama_calibrated $OVERRIDE_RESULTS_DIR
python main.py --config-path=configs/medmcqa --config-name=first_llama_calibrated $OVERRIDE_RESULTS_DIR


OVERRIDE_RESULTS_DIR="results_dir=results_BLR_200 batch_size=100 calibration_size=200 device=cuda:0"

python main.py --config-path=configs/ai2arc_easy --config-name=first_qwen_calibrated $OVERRIDE_RESULTS_DIR
python main.py --config-path=configs/ai2arc_challenge --config-name=first_qwen_calibrated $OVERRIDE_RESULTS_DIR
python main.py --config-path=configs/medqa --config-name=first_qwen_calibrated $OVERRIDE_RESULTS_DIR
python main.py --config-path=configs/medmcqa --config-name=first_qwen_calibrated $OVERRIDE_RESULTS_DIR

python main.py --config-path=configs/ai2arc_easy --config-name=first_llama_calibrated $OVERRIDE_RESULTS_DIR
python main.py --config-path=configs/ai2arc_challenge --config-name=first_llama_calibrated $OVERRIDE_RESULTS_DIR
python main.py --config-path=configs/medqa --config-name=first_llama_calibrated $OVERRIDE_RESULTS_DIR
python main.py --config-path=configs/medmcqa --config-name=first_llama_calibrated $OVERRIDE_RESULTS_DIR


OVERRIDE_RESULTS_DIR="results_dir=results_BLR_500 batch_size=100 calibration_size=500 device=cuda:0"

python main.py --config-path=configs/ai2arc_easy --config-name=first_qwen_calibrated $OVERRIDE_RESULTS_DIR
python main.py --config-path=configs/ai2arc_challenge --config-name=first_qwen_calibrated $OVERRIDE_RESULTS_DIR
python main.py --config-path=configs/medqa --config-name=first_qwen_calibrated $OVERRIDE_RESULTS_DIR
python main.py --config-path=configs/medmcqa --config-name=first_qwen_calibrated $OVERRIDE_RESULTS_DIR

python main.py --config-path=configs/ai2arc_easy --config-name=first_llama_calibrated $OVERRIDE_RESULTS_DIR
python main.py --config-path=configs/ai2arc_challenge --config-name=first_llama_calibrated $OVERRIDE_RESULTS_DIR
python main.py --config-path=configs/medqa --config-name=first_llama_calibrated $OVERRIDE_RESULTS_DIR
python main.py --config-path=configs/medmcqa --config-name=first_llama_calibrated $OVERRIDE_RESULTS_DIR