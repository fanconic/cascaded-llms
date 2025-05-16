
#!/bin/bash

OVERRIDE_RESULTS_DIR="results_dir=results_BLR batch_size=100 precomputed.path=/home/azureuser/caf83/helm/results_3_8_models/run_20250511_131846_first_qwen_arc_easy"
python main.py --config-path=configs/ai2arc_easy --config-name=second_llama
OVERRIDE_RESULTS_DIR="results_dir=results_BLR batch_size=100 precomputed.path=/home/azureuser/caf83/helm/results_3_8_models/run_20250512_073959_first_qwen_arc_challenge"
python main.py --config-path=configs/ai2arc_challenge --config-name=second_llama
OVERRIDE_RESULTS_DIR="results_dir=results_BLR batch_size=100 precomputed.path=/home/azureuser/caf83/helm/results_3_8_models/run_20250512_214151_first_qwen_medqa"
python main.py --config-path=configs/medqa --config-name=second_llama
OVERRIDE_RESULTS_DIR="results_dir=results_BLR batch_size=100 precomputed.path=/home/azureuser/caf83/helm/results_3_8_models/run_20250511_132057_first_qwen_medmcqa"
python main.py --config-path=configs/medmcqa --config-name=second_llama


OVERRIDE_RESULTS_DIR="results_dir=results_BLR batch_size=100 precomputed.path=/home/azureuser/caf83/helm/results_3_8_models/run_20250511_132559_first_llama_arc_easy"
python main.py --config-path=configs/ai2arc_easy --config-name=second_qwen
OVERRIDE_RESULTS_DIR="results_dir=results_BLR batch_size=100 precomputed.path=/home/azureuser/caf83/helm/results_3_8_models/run_20250511_230810_first_llama_arc_challenge"
python main.py --config-path=configs/ai2arc_challenge --config-name=second_qwen
OVERRIDE_RESULTS_DIR="results_dir=results_BLR batch_size=100 precomputed.path=/home/azureuser/caf83/helm/results_3_8_models/run_20250512_063853_first_llama_medqa"
python main.py --config-path=configs/medqa --config-name=second_qwen
OVERRIDE_RESULTS_DIR="results_dir=results_BLR batch_size=100 precomputed.path=/home/azureuser/caf83/helm/results_3_8_models/run_20250511_132402_first_llama_medmcqa"
python main.py --config-path=configs/medmcqa --config-name=second_qwen