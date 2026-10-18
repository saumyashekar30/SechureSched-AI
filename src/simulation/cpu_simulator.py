import pandas as pd
from src.schedulers.fcfs import schedule_fcfs
from src.schedulers.sjf import schedule_sjf
from src.schedulers.round_robin import schedule_rr
from src.schedulers.secure_scheduler import schedule_secure

def run_simulation(df_workload, policy="SecureSched", risk_model=None, rr_quantum=5.0, penalty_lambda=50.0):
    processes = df_workload.copy().to_dict('records')
    for p in processes:
        p['remaining_time'] = p['burst_time']
        p['first_cpu_time'] = None

    completed, ready_queue = [], []
    current_time, idx, n = 0.0, 0, len(processes)

    while len(completed) < n:
        while idx < n and processes[idx]['arrival_time'] <= current_time:
            ready_queue.append(processes[idx])
            idx += 1

        if ready_queue:
            if policy == "FCFS":
                p = schedule_fcfs(ready_queue)
                exec_slice = p['remaining_time']
            elif policy == "SJF":
                p = schedule_sjf(ready_queue)
                exec_slice = p['remaining_time']
            elif policy == "RR":
                p = schedule_rr(ready_queue)
                exec_slice = min(rr_quantum, p['remaining_time'])
            elif policy == "SecureSched":
                p = schedule_secure(ready_queue, risk_model, penalty_lambda)
                exec_slice = p['remaining_time']

            start_time = max(current_time, p['arrival_time'])
            if p['first_cpu_time'] is None:
                p['first_cpu_time'] = start_time

            p['remaining_time'] -= exec_slice
            current_time = start_time + exec_slice

            if p['remaining_time'] <= 0:
                p['completion_time'] = round(current_time, 2)
                p['tat'] = round(p['completion_time'] - p['arrival_time'], 2)
                p['wt'] = round(p['tat'] - p['burst_time'], 2)
                completed.append(p)
            else:
                ready_queue.append(p)
        else:
            if idx < n:
                current_time = processes[idx]['arrival_time']

    return pd.DataFrame(completed)