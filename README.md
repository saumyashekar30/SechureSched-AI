# SecureSched-AI: Security-Aware, AI-Augmented CPU Scheduling

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Model-Isolation%20Forest-orange.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Dataset](https://img.shields.io/badge/Data-Alibaba%20Cluster%20Trace-red.svg)](https://github.com/alibaba/clusterdata)

**SecureSched-AI** is a research framework and discrete-event simulation engine designed to defend CPU scheduling pipelines against adversarial manipulation, priority-gaming exploits, and Denial-of-Service (DoS) starvation attacks. By combining unsupervised machine learning anomaly detection (**Isolation Forest**) with risk-penalized dynamic priority calculation, SecureSched-AI safeguards system throughput and fairness without sacrificing nominal performance.

---

## Table of Contents

- [Overview & Motivation](#overview--motivation)
- [Threat Model & Attack Vector](#threat-model--attack-vector)
- [System Architecture](#system-architecture)
- [How SecureSched Works](#how-securesched-works)
  - [1. Anomaly Detection & Risk Scoring](#1-anomaly-detection--risk-scoring)
  - [2. Dynamic Risk-Penalized Priority](#2-dynamic-risk-penalized-priority)
- [Experimental Results & Benchmarks](#experimental-results--benchmarks)
  - [Comparative Waiting Time (ms)](#comparative-waiting-time-ms)
  - [Key Takeaways](#key-takeaways)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Pipeline](#running-the-pipeline)
  - [Running with Custom Parameters](#running-with-custom-parameters)
- [Core Modules Breakdown](#core-modules-breakdown)

---

## Overview & Motivation

Traditional operating system and distributed cluster schedulers (such as Shortest Job First / SJF, First-Come First-Served / FCFS, and Round Robin / RR) are engineered under the assumption of benign, honest workloads:

- **Shortest Job First (SJF)** minimizes average waiting time in cooperative environments by prioritizing tasks with smaller burst durations.
- **Vulnerability**: In unverified, multi-tenant, or adversarial environments, tasks can self-report falsified short burst times to gain priority, only to execute CPU-intensive payloads for extended durations. This leads to **priority inversion**, **severe thread starvation**, and **catastrophic latency degradation** for legitimate workloads.

**SecureSched-AI** resolves this vulnerability by decoupling process scheduling from naive user-declared metadata, actively evaluating process execution profiles using continuous unsupervised anomaly scoring to detect and deprioritize rogue jobs.

---

## Threat Model & Attack Vector

In this simulation framework, adversarial tasks execute a dual-vector exploit against the scheduler:

1. **Burst Time Under-Declaration (SJF Exploitation)**:
   - Malicious tasks declare a burst time equal to **10%** of their nominal burst requirement:
     $$\text{declared\_burst} = 0.10 \times \text{burst}_{\text{orig}}$$
   - However, during actual CPU execution, they consume **600%** of their nominal burst time:
     $$\text{burst}_{\text{actual}} = 6.0 \times \text{burst}_{\text{orig}}$$
2. **Behavioral System Call Spikes**:
   - While benign processes operate within a nominal syscall frequency range (mean $\mu = 30.0$, $\sigma = 5.0$, bounded between $10.0$ and $50.0$ syscalls/sec), rogue tasks exhibit anomalous syscall rates:
     $$\text{syscall\_rate}_{\text{malicious}} \sim \mathcal{U}(300.0, 500.0)$$

This dual signature allows the machine learning model to isolate adversarial patterns even when declared burst times alone attempt to mimic short, legitimate jobs.

---

## System Architecture

```mermaid
flowchart TD
    A[Alibaba Cluster Workload<br/>batch_instance.csv] --> B[Data Preprocessor<br/>preprocess_alibaba.py]
    B --> C[Clean Baseline Workload<br/>data/processed/processes.csv]
    
    C --> D[Anomaly Injection Module<br/>generate_anomalies.py]
    D -- Inject Attack Rates 0%-20% --> E[Adversarial Workload Stream]
    
    C -. Fit Clean Profile .-> F[Isolation Forest Model<br/>risk_model.py]
    
    E --> G[Discrete-Event CPU Simulator<br/>cpu_simulator.py]
    F -->|Risk Inference R in [0, 1]| G
    
    subgraph Schedulers Evaluated
        G --> H1[FCFS]
        G --> H2[SJF]
        G --> H3[Round Robin]
        G --> H4[SecureSched-AI]
    end
    
    H1 & H2 & H3 & H4 --> I[Metrics & Evaluation Matrix<br/>metrics.py & experiments.py]
    I --> J[Results & Plots<br/>results/raw_results.csv<br/>results/plots/wt_vs_attack_rate.png]
```

---

## How SecureSched Works

### 1. Anomaly Detection & Risk Scoring
The `RiskScoreModel` is based on an `IsolationForest` estimator trained on uncontaminated benign workloads using two feature dimensions:
- `declared_burst`
- `syscall_rate`

Given an incoming process, the raw isolation decision function score $s(x)$ is mapped into a normalized risk score $R \in [0.0, 1.0]$ using an inverted logistic sigmoid transformation:

$$R(x) = \text{clip}\left( \frac{1}{1 + e^{12 \cdot s(x)}}, 0.0, 1.0 \right)$$

- Benign processes receive risk scores close to $0.0$.
- Anomalous processes receive risk scores converging to $1.0$.

### 2. Dynamic Risk-Penalized Priority
Rather than relying solely on `declared_burst`, `SecureSched` schedules the ready queue by sorting processes on an **Effective Priority Score**:

$$\text{Effective Priority} = \text{declared\_burst} + P(R)$$

where the penalty function $P(R)$ is governed by penalty multiplier $\lambda$ (default: $\lambda = 500.0$):

$$P(R) = \begin{cases} 
10 \cdot \lambda \cdot R, & \text{if } R > 0.5 \quad \text{(Severe Penalty for Confirmed Threats)} \\
\lambda \cdot R, & \text{if } R \le 0.5 
\end{cases}$$

Legitimate tasks preserve low priority scores and are executed promptly, while malicious tasks are pushed to the rear of the queue, insulating benign tasks from CPU starvation.

---

## Experimental Results & Benchmarks

Simulations were performed against real-world Alibaba Cloud trace samples with attack penetration rates spanning $0\%$ to $20\%$.

### Comparative Waiting Time (ms)

| Malicious Task Injection Rate (%) | FCFS (ms) | SJF (ms) | Round Robin (ms) | **SecureSched-AI (ms)** | SJF Latency Penalty vs. SecureSched |
|:--------------------------------:|:---------:|:--------:|:----------------:|:-----------------------:|:----------------------------------:|
| **0% (Nominal)**                 | 547.95    | **526.10** | 1049.10          | 537.95                  | Baseline (+2.2% overhead)          |
| **5%**                           | 842.45    | 825.30   | 1052.20          | **537.05**              | +53.7% higher latency              |
| **10%**                          | 869.95    | 1073.05  | 1083.15          | **552.35**              | +94.3% higher latency              |
| **15%**                          | 923.95    | 1302.35  | 1141.80          | **580.85**              | +124.2% higher latency             |
| **20% (Heavy Attack)**           | 1180.45   | 1526.60  | 1228.95          | **623.05**              | **+145.0% higher latency**         |

### Key Takeaways
1. **Resilience to Starvation**: Under a 20% injection attack, traditional SJF waiting time degrades by **+190.2%** (from 526.10 ms to 1526.60 ms) because rogue tasks repeatedly preempt legitimate processes.
2. **Minimal Nominal Overhead**: In an uncontaminated environment (0% attack), SecureSched incurs only negligible overhead (+2.2% over ideal SJF, 537.95 ms vs 526.10 ms).
3. **Attack Resistance**: At the 20% attack tier, SecureSched outperforms standard SJF by **59.2%** and FCFS by **47.2%**, preserving low latency and workload stability.

Benchmark plots are automatically generated and saved to [`results/plots/wt_vs_attack_rate.png`](file:///c:/Users/Saumya/SecureSched_Research/results/plots/wt_vs_attack_rate.png).

---

## Repository Structure

```
SecureSched_Research/
├── data/
│   ├── raw/                           # Raw trace archives from Alibaba Cluster Data
│   │   ├── batch_instance.csv
│   │   ├── batch_task.csv
│   │   └── container_usage.csv
│   └── processed/
│       └── processes.csv              # Extracted and structured benchmark workload
├── models/                            # Checkpoints and serialized model outputs
├── results/
│   ├── plots/
│   │   └── wt_vs_attack_rate.png      # Comparative resilience curve plot
│   ├── tables/                        # Formatted experiment tables
│   └── raw_results.csv                # Metrics (avg_wt, avg_tat, jain_fairness)
├── src/
│   ├── anomaly/
│   │   └── generate_anomalies.py      # Malicious burst gaming and syscall spike generator
│   ├── evaluation/
│   │   ├── experiments.py             # Evaluation harness, multi-policy benchmarks, plot generation
│   │   └── metrics.py                 # Waiting time, turnaround time, and Jain's Fairness index
│   ├── processing/
│   │   └── preprocess_alibaba.py      # Alibaba dataset parsing, cleaning, and sampling
│   ├── schedulers/
│   │   ├── fcfs.py                    # First-Come First-Served scheduler
│   │   ├── round_robin.py             # Round Robin scheduler
│   │   ├── secure_scheduler.py        # SecureSched risk-penalized scheduling algorithm
│   │   └── sjf.py                     # Shortest Job First scheduler
│   ├── security/
│   │   └── risk_model.py              # Isolation Forest anomaly detection & risk scoring
│   ├── simulation/
│   │   └── cpu_simulator.py           # Discrete-event multi-policy CPU simulator
│   └── utils/
│       └── config.py                  # Global configurations, file paths, and hyperparameters
├── main.py                            # Main execution pipeline entry point
├── pipeline.ipynb                     # Jupyter notebook for interactive experiments
├── requirements.txt                   # Project dependencies
└── README.md                          # Comprehensive documentation
```

---

## Getting Started

### Prerequisites
- Python 3.10 or higher
- Git

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/saumyashekar30/SechureSched-AI.git
   cd SechureSched-AI
   ```

2. Create and activate a virtual environment:
   ```bash
   # Windows (PowerShell)
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # Linux / macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Pipeline

To run the complete benchmark suite across all policies (`FCFS`, `SJF`, `RR`, and `SecureSched`) and attack rates ($0\%$ to $20\%$):

```bash
python main.py
```

**Expected Output:**
```text
=== Starting SecureSched-AI Pipeline Execution ===
--> Loading existing processed workload from .../data/processed/processes.csv

--> Running Evaluation Matrix across Attack Scenarios...
Experiments completed! Results saved to .../results

=== Experiment Summary Table (Average Waiting Time ms) ===
policy          FCFS       RR      SJF  SecureSched
attack_rate                                        
0             547.95  1049.10   526.10       537.95
5             842.45  1052.20   825.30       537.05
10            869.95  1083.15  1073.05       552.35
15            923.95  1141.80  1302.35       580.85
20           1180.45  1228.95  1526.60       623.05
```

Generated plots will be stored in `results/plots/wt_vs_attack_rate.png` and numerical outputs in `results/raw_results.csv`.

### Running with Custom Parameters
You can adjust experiment parameters in [`src/utils/config.py`](file:///c:/Users/Saumya/SecureSched_Research/src/utils/config.py):
- `ATTACK_RATES`: Percentage range of injected malicious tasks (e.g. `[0.0, 0.05, 0.10, 0.15, 0.20, 0.30]`).
- `RISK_LAMBDA`: Penalty weight multiplier for high-risk scores (default: `500.0`).
- `RR_QUANTUM`: Time quantum for Round Robin scheduling (default: `5.0`).

---

## Core Modules Breakdown

| Module | File | Key Functions / Classes | Responsibility |
|---|---|---|---|
| **Security Risk Model** | [`src/security/risk_model.py`](file:///c:/Users/Saumya/SecureSched_Research/src/security/risk_model.py) | `RiskScoreModel` | Fits `IsolationForest` on clean workloads and derives normalized risk scores $R \in [0, 1]$. |
| **Secure Scheduler** | [`src/schedulers/secure_scheduler.py`](file:///c:/Users/Saumya/SecureSched_Research/src/schedulers/secure_scheduler.py) | `schedule_secure()` | Calculates risk-penalized effective priorities and dispatches the optimal job. |
| **CPU Simulator** | [`src/simulation/cpu_simulator.py`](file:///c:/Users/Saumya/SecureSched_Research/src/simulation/cpu_simulator.py) | `run_simulation()` | Discrete-event simulation tracking arrival, execution slices, turnaround, and waiting times. |
| **Anomaly Generator** | [`src/anomaly/generate_anomalies.py`](file:///c:/Users/Saumya/SecureSched_Research/src/anomaly/generate_anomalies.py) | `inject_anomalies()` | Perturbs declared burst and injects anomalous syscall rates into selected tasks. |
| **Evaluation Suite** | [`src/evaluation/experiments.py`](file:///c:/Users/Saumya/SecureSched_Research/src/evaluation/experiments.py) | `run_experiment_matrix()` | Iterates through attack matrix, records metrics, and produces comparison figures. |
| **Metrics** | [`src/evaluation/metrics.py`](file:///c:/Users/Saumya/SecureSched_Research/src/evaluation/metrics.py) | `calculate_metrics()` | Computes Average Waiting Time, Turnaround Time, and Jain's Fairness Index. |
| **Data Ingestion** | [`src/processing/preprocess_alibaba.py`](file:///c:/Users/Saumya/SecureSched_Research/src/processing/preprocess_alibaba.py) | `preprocess_data()` | Cleans raw Alibaba trace data, samples valid instances, and synthesizes baseline features. |

---

## PPO Scheduler (Deep Reinforcement Learning extension)

A Maskable PPO (Stable-Baselines3 / sb3-contrib) scheduling policy is available as an **additional** method next to
FCFS, SJF, Round Robin and SecureSched. It uses the Isolation-Forest risk score as an observation and does not
change any existing module or result (`results/raw_results.csv` is untouched; PPO output goes to `results/ppo/`).
The original pipeline (`python main.py`) still runs without the PPO dependencies.

### Setup
```bash
python3 -m venv .venv && source .venv/bin/activate      # recommended (required with Homebrew Python)
python -m pip install -r requirements-ppo.txt
python -m pytest tests -q                               # 12 tests: environment, metrics, save/load, baselines unchanged
```

### Train and evaluate
```bash
for s in 42 43 44; do python -m src.ppo.train --seed $s --out-dir results/ppo/models/seed_$s; done   # ~6 min each
python -m src.ppo.evaluate                              # ~5 min; tables and plots in results/ppo/
```
Training options: `--timesteps --learning-rate --n-steps --batch-size --gamma --clip-range`. The trained models,
plots and raw result files are generated by these commands and may not be committed to the repository.

### What it does
- **Environment** (`src/ppo/sched_env.py`): non-preemptive single CPU; the agent picks one of 16 candidate processes
  (invalid slots are masked). It observes declared burst, waiting time and risk score, never the true burst or labels.
- **Reward:** normalised waiting time, aging-weighted delay (fairness), a penalty for dispatching flagged processes
  ahead of unflagged ones, and idle time.
- **Evaluation:** Experiment A reruns the committed seed-42 workload (baselines match `results/raw_results.csv`
  exactly); Experiment B uses held-out Alibaba trace windows with several injection and training seeds.

### Measured results (held-out windows, mean waiting time, PPO = mean of 3 training seeds)

| attack % | FCFS | SJF | RR | SecureSched | PPO |
|---|---|---|---|---|---|
| 0 | 838 | 778 | 1527 | 796 | 809 |
| 10 | 1276 | 1581 | 1618 | 1089 | 1159 |
| 20 | 1674 | 2296 | 1861 | 1409 | 1576 |

PPO beats FCFS, SJF and Round Robin under attack but does **not** outperform the heuristic SecureSched scheduler
(higher mean waiting time and a higher security-violation rate at every ratio). Training reward was still rising at
150k steps and the reward weights were not tuned, so this is not a final verdict on PPO. Results from other library
versions or machines may differ slightly.

Full MDP formulation, reward, metric definitions, methodology and limitations: [`docs/PPO.md`](docs/PPO.md).