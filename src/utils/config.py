import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Path definitions
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")

# Data File Paths
BATCH_TASK_PATH = os.path.join(RAW_DATA_DIR, "batch_task.csv")
BATCH_INSTANCE_PATH = os.path.join(RAW_DATA_DIR, "batch_instance.csv")
CONTAINER_USAGE_PATH = os.path.join(RAW_DATA_DIR, "container_usage.csv")
PROCESSED_PROCESSES_PATH = os.path.join(PROCESSED_DATA_DIR, "processes.csv")

# Experiment Configurations
ATTACK_RATES = [0.0, 0.05, 0.10, 0.15, 0.20]
RISK_LAMBDA = 500.0  # Priority penalty multiplier for high risk scores
RR_QUANTUM = 5.0
