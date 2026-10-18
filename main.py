import os
import sys

# Ensure project root is in python path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
from src.utils.config import PROCESSED_PROCESSES_PATH
# Updated import path to match your folder name 'processing'
from src.processing.preprocess_alibaba import preprocess_data
from src.evaluation.experiments import run_experiment_matrix

def main():
    print("=== Starting SecureSched-AI Pipeline Execution ===")
    
    if not os.path.exists(PROCESSED_PROCESSES_PATH):
        print("--> Processed workload not found. Running preprocessor...")
        df_clean = preprocess_data()
    else:
        print(f"--> Loading existing processed workload from {PROCESSED_PROCESSES_PATH}")
        df_clean = pd.read_csv(PROCESSED_PROCESSES_PATH)
        
    print("\n--> Running Evaluation Matrix across Attack Scenarios...")
    results_df = run_experiment_matrix(df_clean)
    
    print("\n=== Experiment Summary Table (Average Waiting Time ms) ===")
    print(results_df.pivot(index='attack_rate', columns='policy', values='avg_wt'))

if __name__ == "__main__":
    main()