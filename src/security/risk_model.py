import numpy as np
from sklearn.ensemble import IsolationForest

class RiskScoreModel:
    def __init__(self, contamination=0.1, random_state=42):
        # Increased contamination estimate for better decision boundary
        self.model = IsolationForest(contamination=contamination, random_state=random_state)
        
    def fit(self, df_clean):
        features = df_clean[['declared_burst', 'syscall_rate']].values
        self.model.fit(features)
        
    def predict_risk(self, declared_burst, syscall_rate):
        # Convert decision function into normalized risk score: high anomaly = score close to 1.0
        # raw score: lower means more abnormal
        raw_score = self.model.decision_function([[declared_burst, syscall_rate]])[0]
        
        # Invert and scale decision function output
        risk_score = 1.0 / (1.0 + np.exp(raw_score * 12.0))
        return float(np.clip(risk_score, 0.0, 1.0))