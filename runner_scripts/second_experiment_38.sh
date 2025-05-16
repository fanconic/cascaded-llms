
#!/bin/bash

OVERRIDE_RESULTS_DIR="results_dir=results_online_38 "

python main.py --config-path=configs/ai2arc_easy --config-name=second_llama $OVERRIDE_RESULTS_DIR
python main.py --config-path=configs/ai2arc_challenge --config-name=second_llama $OVERRIDE_RESULTS_DIR
python main.py --config-path=configs/medqa --config-name=second_llama $OVERRIDE_RESULTS_DIR
python main.py --config-path=configs/medmcqa --config-name=second_llama $OVERRIDE_RESULTS_DIR

python main.py --config-path=configs/ai2arc_easy --config-name=second_qwen $OVERRIDE_RESULTS_DIR
python main.py --config-path=configs/ai2arc_challenge --config-name=second_qwen $OVERRIDE_RESULTS_DIR
python main.py --config-path=configs/medqa --config-name=second_qwen $OVERRIDE_RESULTS_DIR
python main.py --config-path=configs/medmcqa --config-name=second_qwen $OVERRIDE_RESULTS_DIR