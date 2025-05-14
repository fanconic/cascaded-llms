
#!/bin/bash

OVERRIDE_RESULTS_DIR="results_dir=results_BLR batch_size=100"

python main.py --config-path=configs/ai2arc_easy --config-name=second_llama && python main.py --config-path=configs/ai2arc_challenge --config-name=second_llama

python main.py --config-path=configs/medqa --config-name=second_llama && python main.py --config-path=configs/medmcqa --config-name=second_llama

python main.py --config-path=configs/ai2arc_easy --config-name=second_qwen && python main.py --config-path=configs/ai2arc_challenge --config-name=second_qwen

python main.py --config-path=configs/medqa --config-name=second_qwen && python main.py --config-path=configs/medmcqa --config-name=second_qwen