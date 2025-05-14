import pandas as pd
import os

path = "/home/azureuser/caf83/helm/followup_results/run_20250506_231520_first_llama_medmcqa"
result_path  = "/home/azureuser/caf83/helm/followup_results/run_20250506_231520_first_llama_medmcqa_full"
prependix_path = "/home/azureuser/caf83/helm/followup_results/run_20250508_094841_first_llama_medmcqa_preprendix"

for file, file_prependix in zip(sorted(os.listdir(path)), sorted(os.listdir(prependix_path))):
    if "results" in file and "results" in file_prependix:
        data = pd.read_csv(os.path.join(path, file), engine='python').reset_index()
        data_preprendix = pd.read_csv(os.path.join(prependix_path, file_prependix), engine='python').reset_index()
        
        columns = list(data.columns)[1:]
        data = data.drop(["system_risk_static"], axis=1)
        data.columns = columns
        
        data_merge = pd.concat([data_preprendix, data]).drop(["index"], axis=1)
        
        data_merge.to_csv(os.path.join(result_path, file), index=False)