import os
import numpy as np
import pandas as pd
from src.utils.config import (
    BATCH_INSTANCE_PATH, 
    CONTAINER_USAGE_PATH, 
    PROCESSED_PROCESSES_PATH,
    PROCESSED_DATA_DIR
)

def preprocess_data(sample_size=1000):
    print("Loading raw Alibaba dataset files...")
    
    # Read instance data
    instance_df = pd.read_csv(
        BATCH_INSTANCE_PATH, 
        header=None, 
        nrows=50000, 
        usecols=[0, 1, 2, 3, 4, 5]
    )
    instance_df.columns = ['start_time', 'end_time', 'job_id', 'task_id', 'instance_id', 'status']
    
    # Clean timing metrics
    instance_df = instance_df.dropna(subset=['start_time', 'end_time']).copy()
    instance_df['burst_time'] = instance_df['end_time'] - instance_df['start_time']
    valid_instances = instance_df[instance_df['burst_time'] > 0].copy()
    
    # Sample workload
    sampled = valid_instances.sample(n=min(sample_size, len(valid_instances)), random_state=42).copy()
    sampled = sampled.sort_values(by='start_time').reset_index(drop=True)
    
    # Generate tightly bounded nominal syscall rates for benign processes (e.g. 15.0 to 45.0)
    np.random.seed(42)
    nominal_syscalls = np.random.normal(loc=30.0, scale=5.0, size=len(sampled)).clip(10.0, 50.0)
    
    processes_df = pd.DataFrame({
        'pid': range(1, len(sampled) + 1),
        'arrival_time': np.round(sampled['start_time'].values - sampled['start_time'].min(), 2),
        'burst_time': np.round(sampled['burst_time'].values, 2),
        'declared_burst': np.round(sampled['burst_time'].values, 2),
        'syscall_rate': np.round(nominal_syscalls, 2),
        'is_malicious': 0
    })
    
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    processes_df.to_csv(PROCESSED_PROCESSES_PATH, index=False)
    print(f"Preprocessed workload successfully saved to {PROCESSED_PROCESSES_PATH}")
    return processes_df

if __name__ == "__main__":
    preprocess_data()