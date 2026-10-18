import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ..utils.config import ATTACK_RATES, RESULTS_DIR, PLOTS_DIR, TABLES_DIR, RISK_LAMBDA, RR_QUANTUM
from ..anomaly.generate_anomalies import inject_anomalies
from ..security.risk_model import RiskScoreModel
from ..simulation.cpu_simulator import run_simulation
from .metrics import calculate_metrics

def run_experiment_matrix(df_clean):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)
    
    risk_model = RiskScoreModel()
    risk_model.fit(df_clean)
    
    policies = ["FCFS", "SJF", "RR", "SecureSched"]
    all_results = []
    
    for rate in ATTACK_RATES:
        df_attack = inject_anomalies(df_clean, rate)
        for pol in policies:
            df_res = run_simulation(
                df_attack, 
                policy=pol, 
                risk_model=risk_model, 
                rr_quantum=RR_QUANTUM, 
                penalty_lambda=RISK_LAMBDA
            )
            metrics = calculate_metrics(df_res)
            all_results.append({
                'attack_rate': int(rate * 100),
                'policy': pol,
                'avg_wt': metrics['avg_wt'],
                'avg_tat': metrics['avg_tat'],
                'jain_fairness': metrics['jain_fairness']
            })
            
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(RESULTS_DIR, "raw_results.csv"), index=False)
    
    # Generate visualization
    plt.figure(figsize=(9, 5))
    sns.lineplot(data=results_df, x='attack_rate', y='avg_wt', hue='policy', marker='o', linewidth=2.5)
    plt.title("Security Resilience: Average Waiting Time under Attack", fontsize=12, fontweight='bold')
    plt.xlabel("Percentage of Injected Malicious Tasks (%)")
    plt.ylabel("Average Waiting Time (ms)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "wt_vs_attack_rate.png"), dpi=300)
    plt.close()
    
    print(f"Experiments completed! Results saved to {RESULTS_DIR}")
    return results_df