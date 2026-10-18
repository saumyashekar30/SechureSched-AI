def schedule_secure(ready_queue, risk_model, penalty_lambda=500.0):
    for p in ready_queue:
        if 'risk_score' not in p or p['risk_score'] is None:
            r_score = risk_model.predict_risk(p['declared_burst'], p['syscall_rate'])
            p['risk_score'] = r_score
            
            # If risk is high, penalize severely using multiplier
            if r_score > 0.5:
                p['effective_priority'] = p['declared_burst'] + (penalty_lambda * r_score * 10)
            else:
                p['effective_priority'] = p['declared_burst'] + (penalty_lambda * r_score)
            
    ready_queue.sort(key=lambda x: x['effective_priority'])
    return ready_queue.pop(0)