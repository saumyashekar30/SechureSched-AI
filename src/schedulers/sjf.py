def schedule_sjf(ready_queue):
    ready_queue.sort(key=lambda x: x['declared_burst'])
    return ready_queue.pop(0)