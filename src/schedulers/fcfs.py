def schedule_fcfs(ready_queue):
    ready_queue.sort(key=lambda x: x['arrival_time'])
    return ready_queue.pop(0)