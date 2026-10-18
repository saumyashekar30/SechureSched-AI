import numpy as np

def inject_anomalies(df, attack_ratio, seed=42):
    np.random.seed(seed)
    df_mod = df.copy()
    num_malicious = int(len(df_mod) * attack_ratio)
    
    if num_malicious == 0:
        return df_mod
        
    malicious_indices = np.random.choice(df_mod.index, size=num_malicious, replace=False)
    
    for idx in malicious_indices:
        df_mod.loc[idx, 'is_malicious'] = 1
        orig_burst = df_mod.loc[idx, 'burst_time']
        # Game the system: Declare 10% of true burst time, actual burst is 6x longer
        df_mod.loc[idx, 'declared_burst'] = round(orig_burst * 0.1, 2)
        df_mod.loc[idx, 'burst_time'] = round(orig_burst * 6.0, 2)
        # Spike system call activity
        df_mod.loc[idx, 'syscall_rate'] = round(np.random.uniform(300.0, 500.0), 2)
        
    return df_mod