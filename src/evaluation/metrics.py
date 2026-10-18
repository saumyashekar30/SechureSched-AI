import numpy as np

def calculate_metrics(df_results):
    avg_wt = df_results['wt'].mean()
    avg_tat = df_results['tat'].mean()
    
    # Calculate Jain's Fairness Index on Waiting Times
    wts = df_results['wt'].values
    n = len(wts)
    jain_index = (np.sum(wts) ** 2) / (n * np.sum(wts ** 2)) if np.sum(wts ** 2) > 0 else 1.0
    
    return {
        'avg_wt': round(avg_wt, 3),
        'avg_tat': round(avg_tat, 3),
        'jain_fairness': round(jain_index, 4)
    }